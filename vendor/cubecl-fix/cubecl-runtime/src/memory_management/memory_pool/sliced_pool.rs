use crate::{
    memory_management::{
        BytesFormat, ManagedMemoryHandle, MemoryLocation, MemoryPoolKind, MemoryPoolReport,
        MemoryUsage,
        memory_pool::{MemoryPage, MemoryPool, PageMapping, Slice},
    },
    server::IoError,
    storage::StorageId,
};
use alloc::vec::Vec;
use core::fmt::Display;
use cubecl_environment::backtrace::BackTrace;

pub struct SlicedPool {
    /// Slot-preserving page storage: freed pages leave a hole (`None`) that
    /// future allocations reuse, so a page's index is **permanent for the
    /// pool's lifetime**. Descriptors cache their page index; cleanup no
    /// longer renumbers survivors, so cached indices can never go stale.
    pages: Vec<Option<(MemoryPage, StorageId)>>,
    /// Indices of holes in `pages`, popped by allocation before growing.
    free_slots: Vec<usize>,
    page_size: u64,
    alignment: u64,
    max_alloc_size: u64,
    location_base: MemoryLocation,
    /// Max number of pages (`floor(max_pool_size / page_size)`).
    /// `None` keeps unbounded growth.
    max_pages: Option<u16>,
    /// The most pages ever held at once. Pages are only freed by an explicit
    /// cleanup, so this is the pool's true high-water mark whenever one runs
    /// mid-workload.
    pages_peak: u64,
    /// Occupied slots (`pages` entries that are `Some`).
    pages_live: usize,
    /// The largest allocation ever served, in requested (pre-padding) bytes.
    largest_alloc: u64,
}

impl SlicedPool {
    pub fn new(
        page_size: u64,
        max_slice_size: u64,
        alignment: u64,
        pool_pos: u8,
        max_pool_size: Option<u64>,
    ) -> Self {
        // A budget smaller than one page shrinks the page to the
        // (alignment-rounded-down) budget, so the cap is honored rather than
        // exceeded by a single page. A budget below the alignment can't fit
        // even the smallest page the device allows, so it yields zero pages:
        // allocations error instead of overshooting the cap.
        let (page_size, max_pages) = match max_pool_size {
            Some(cap) => {
                let page_size = if cap < page_size {
                    (cap / alignment * alignment).max(alignment)
                } else {
                    page_size
                };
                let max_pages = (cap / page_size).min(u16::MAX as u64) as u16;
                (page_size, Some(max_pages))
            }
            None => (page_size, None),
        };

        Self {
            pages: Vec::new(),
            free_slots: Vec::new(),
            page_size,
            alignment,
            max_alloc_size: max_slice_size.min(page_size),
            location_base: MemoryLocation::new(pool_pos, 0, 0),
            max_pages,
            pages_peak: 0,
            pages_live: 0,
            largest_alloc: 0,
        }
    }

    /// A structured snapshot of the pool: shape, usage, high-water marks.
    pub(crate) fn report(&self) -> MemoryPoolReport {
        MemoryPoolReport {
            kind: MemoryPoolKind::Sliced {
                page_size: self.page_size,
                max_slice_size: self.max_alloc_size,
                max_pool_size: self.max_pages.map(|pages| pages as u64 * self.page_size),
            },
            usage: self.get_memory_usage(),
            pages: self.pages_live as u64,
            pages_peak: self.pages_peak,
            pages_unmapped: self
                .pages
                .iter()
                .flatten()
                .filter(|(page, _)| !page.is_mapped())
                .count() as u64,
            largest_alloc: self.largest_alloc,
        }
    }

    /// Allocate a new page and return its index.
    fn alloc_page<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        mapping: PageMapping,
    ) -> Result<usize, IoError> {
        // Reuse a freed slot when one exists: a page index, once handed out
        // inside a descriptor location, is only ever retired by the owner —
        // never shifted by cleanup and never renumbered.
        let idx = match self.free_slots.pop() {
            Some(idx) => idx,
            None => {
                self.pages.push(None);
                self.pages.len() - 1
            }
        };

        let mut location_base = self.location_base;
        location_base.page = idx as u16;

        // A lazy page gets a minted id with no device memory behind it: it
        // carves, coalesces and counts toward the high-water exactly like a
        // real one, and is rebound to a real allocation on first resolution
        // (`materialize`).
        let storage = mapping.storage_handle(storage, self.page_size)?;
        let page = MemoryPage::new(storage, self.alignment, location_base, mapping);
        let storage_id = page.storage_id();
        self.pages[idx] = Some((page, storage_id));
        self.pages_live += 1;
        self.pages_peak = self.pages_peak.max(self.pages_live as u64);

        Ok(idx)
    }
}

impl MemoryPool for SlicedPool {
    fn accept(&self, size: u64) -> bool {
        self.max_alloc_size >= size
            ||
            // If the size is close to the page size so it doesn't create much fragmentation with
            // unused space. Only for unbounded pools: a hard-capped pool is a budget for the
            // allocations `max_slice_size` routes to it, and near-page-size strays would exhaust
            // it (e.g. a small metadata pool whose page size matches an upload staging chunk).
            (self.max_pages.is_none()
                && match self.page_size.checked_sub(size) {
                    Some(diff) => diff * 5 < self.page_size, // 20 % unused space is the max allowed.
                    None => false,
                })
    }

    fn find(&self, binding: &super::ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let page_index = binding.descriptor().page();
        let (page, _) = self
            .pages
            .get(page_index)
            .and_then(|slot| slot.as_ref())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Memory page {page_index} doesn't exist").into(),
            })?;
        let slice = page.find(binding)?;
        // Page indices are permanent but reusable, and slices within a page
        // are renumbered by coalescing: a binding whose cached location was
        // retired must surface as `NotFound`, never as another slice's data.
        if slice.handle.descriptor().id != binding.descriptor().id {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!("Stale binding for page {page_index}").into(),
            });
        }
        Ok(slice)
    }

    fn try_reserve(&mut self, size: u64) -> Option<super::ManagedMemoryHandle> {
        for (page, _) in self.pages.iter_mut().flatten() {
            page.coalesce();
            if let Some(handle) = page.try_reserve(size) {
                self.largest_alloc = self.largest_alloc.max(size);
                return Some(handle);
            }
        }

        None
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        mapping: PageMapping,
    ) -> Result<super::ManagedMemoryHandle, crate::server::IoError> {
        // `alloc` is only called after `try_reserve` coalesced every page and
        // found no fit, so hitting the cap here means the working set truly
        // exceeds the budget.
        if let Some(max_pages) = self.max_pages
            && self.pages_live >= max_pages as usize
        {
            return Err(IoError::PoolCapacityExceeded {
                size,
                capacity: max_pages as u64 * self.page_size,
                in_use: self.get_memory_usage().bytes_in_use,
                backtrace: BackTrace::capture(),
            });
        }

        let index = self.alloc_page(storage, mapping)?;
        let (page, _) = self.pages[index].as_mut().expect("just-allocated page");
        let returned = page.try_reserve(size);
        self.largest_alloc = self.largest_alloc.max(size);

        Ok(returned.expect("effective_size to be smaller than page_size"))
    }

    fn materialize<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        binding: &super::ManagedMemoryBinding,
    ) -> Result<(), IoError> {
        let page_index = binding.descriptor().page();
        // An out-of-range page is `find`'s error to report, not ours.
        let Some((page, id)) = self
            .pages
            .get_mut(page_index)
            .and_then(|slot| slot.as_mut())
        else {
            return Ok(());
        };
        if page.is_mapped() {
            return Ok(());
        }
        // So is a stale location whose page index a later cleanup reassigned:
        // it names a page this binding has no claim on, and backing that page
        // would allocate device memory for an allocation nobody asked to
        // resolve — the opposite of what a dry run is for.
        let claimed = page
            .find(binding)
            .is_ok_and(|slice| slice.handle.descriptor() == binding.descriptor());
        if !claimed {
            return Ok(());
        }

        // The virtual carving *is* the layout: allocate the page for real and
        // rebind — every slice keeps its offset, the minted id ceases to
        // exist (it never reached the driver).
        let real = storage
            .alloc(self.page_size)
            .map_err(|err| IoError::StorageMappingFailed {
                size: self.page_size,
                source: alloc::boxed::Box::new(err),
                backtrace: BackTrace::capture(),
            })?;
        page.rebind_storage(real.id);
        *id = real.id;

        Ok(())
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        let mut usage = MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        };

        for (page, _) in self.pages.iter().flatten() {
            let current = page.memory_usage();
            usage = usage.combine(current);
        }

        usage
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn cleanup<Storage: crate::storage::ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        _alloc_nr: u64,
        explicit: bool,
    ) {
        if !explicit {
            return;
        }

        // Pages are freed in place: the slot becomes a hole for future
        // allocations to reuse, and every surviving page keeps its index.
        // Descriptors aliasing a page therefore never observe a stale page
        // index after a cleanup. (Slices *within* a page are still renumbered
        // by `coalesce`; stale slice bindings surface from `find`.)
        for idx in 0..self.pages.len() {
            let dealloc_id = match &mut self.pages[idx] {
                Some((page, id)) => {
                    page.coalesce();
                    let summary = page.summary(false);

                    if summary.amount_free != summary.amount_total {
                        continue;
                    }
                    // An unmapped page has nothing behind its minted id; handing
                    // it to the driver's deferred-free queue would be garbage.
                    page.is_mapped().then_some(*id)
                }
                None => continue,
            };

            if let Some(id) = dealloc_id {
                storage.dealloc(id);
            }
            self.pages[idx] = None;
            self.free_slots.push(idx);
            self.pages_live -= 1;
        }
    }

    /// Binds a user defined [`ManagedMemoryHandle`] to a slice in this memory pool.
    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let (page, _) = self
            .pages
            .get_mut(reserved.descriptor().page())
            .and_then(|slot| slot.as_mut())
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: alloc::format!(
                    "Memory page {} doesn't exist",
                    reserved.descriptor().page()
                )
                .into(),
            })?;

        page.bind(reserved, assigned, cursor)?;

        Ok(())
    }
}

impl Display for SlicedPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.pages.is_empty() {
            return Ok(());
        }

        f.write_fmt(format_args!(
            " - Sliced Pool page_size={} max_alloc_size={}",
            BytesFormat::new(self.page_size),
            BytesFormat::new(self.max_alloc_size)
        ))?;
        if let Some(max_pages) = self.max_pages {
            f.write_fmt(format_args!(
                " max_pool_size={}",
                BytesFormat::new(max_pages as u64 * self.page_size)
            ))?;
        }
        f.write_str("\n")?;

        for (page, id) in self.pages.iter().flatten() {
            let summary = page.summary(false);
            f.write_fmt(format_args!(
                "   - Page {id} num_slices={} =>",
                summary.num_total
            ))?;

            let size_free = BytesFormat::new(summary.amount_free);
            let size_full = BytesFormat::new(summary.amount_full);
            let size_total = BytesFormat::new(summary.amount_total);

            f.write_fmt(format_args!(
                " {size_free} free - {size_full} full - {size_total} total\n"
            ))?;
        }

        f.write_fmt(format_args!("\n{}\n", self.get_memory_usage()))?;

        Ok(())
    }
}

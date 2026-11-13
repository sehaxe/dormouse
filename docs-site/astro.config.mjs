import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The docs live under the landing page: sehaxe.github.io/dormouse/ is the
// product page, sehaxe.github.io/dormouse/docs/ is this site.
export default defineConfig({
  site: 'https://sehaxe.github.io',
  base: '/dormouse/docs',
  outDir: './dist',
  integrations: [
    starlight({
      title: 'dormouse',
      description:
        'A byte-level language model trained on one GPU: 1.19 bits per byte, 82 MB ternary.',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/sehaxe/dormouse' },
      ],
      customCss: ['./src/styles/theme.css'],
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
      sidebar: [
        { label: 'Start here', items: [{ slug: 'index' }, { slug: 'handbook' }, { slug: 'quick-start' }] },
        {
          label: 'How it works',
          items: [{ slug: 'architecture' }, { slug: 'formats' }, { slug: 'measurements' }],
        },
        { label: 'What is next', items: [{ slug: 'roadmap' }] },
        { label: 'Archive', items: [{ slug: 'archive' }] },
      ],
      head: [
        // the same paper/ink palette the landing page uses
        { tag: 'meta', attrs: { name: 'theme-color', content: '#f6f3ec' } },
      ],
    }),
  ],
});

import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { z } from 'astro:content';

// `canonical` is the repo-relative path of the file this page was generated
// from. It powers the "Edit page" link (astro.config.mjs sends it to the
// canonical file, never to the generated one) and it is how a reader knows
// which file is the real one.
export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({
			extend: z.object({
				canonical: z.string().optional(),
				ingested_from: z.string().optional(),
			}),
		}),
	}),
};
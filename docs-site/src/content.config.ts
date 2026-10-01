import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

// `canonical` is the repo-relative path of the file this page was generated
// from. It powers the "Edit page" link (astro.config.mjs sends it to the
// canonical file, never to the generated one) and it is how a reader knows
// which file is the real one.
export const collections = {
	// Starlight reads an `i18n` collection for UI-string overrides. This site
	// translates nothing, but an absent collection warns on every build, so the
	// one locale that exists is declared with no overrides in it.
	i18n: defineCollection({
		loader: glob({ pattern: '**/ui.{json,js}', base: './src/content/i18n' }),
		schema: i18nSchema(),
	}),
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
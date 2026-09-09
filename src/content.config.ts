import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const reviews = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/reviews' }),
  schema: z.object({
    title: z.string(),
    paper: z.string().optional(),
    venue: z.string().optional(),
    authors: z.string().optional(),
    link: z.string().optional(),
    claim: z.string(),
    take: z.string().optional(),
    tags: z.array(z.string()).default([]),
    tier: z.enum(['main', 'basic']).default('main'),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
  }),
});

const knowledge = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/knowledge' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    tags: z.array(z.string()).default([]),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
  }),
});

const projects = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    role: z.string().optional(),
    context: z.string().optional(),
    period: z.string().optional(),
    stack: z.string().optional(),
    tags: z.array(z.string()).default([]),
    date: z.coerce.date(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { reviews, knowledge, projects };

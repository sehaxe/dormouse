---
title: dormouse
description: A byte-level language-model trainer that fits on one 16 GB workstation — the measurements that justify it and the ones that refute it.
template: splash
editUrl: false
hero:
  tagline: The knowledge base — every document in this repository, indexed, cross-linked, and generated from the canonical files at build time. Nothing here is a copy.
  actions:
    - text: What is true right now
      link: /start-here/status/
      icon: right-arrow
      variant: primary
    - text: How the model works
      link: /architecture/model/
      icon: right-arrow
      variant: secondary
    - text: What was retracted
      link: /archive/broken/
      icon: open-book
      variant: secondary
---

## A knowledge base with a retraction policy

Eight sections, one interface each: **Start here** (what this is and what is
measured), **Architecture**, **Protocols**, **ADR**, **Research**, **Reviews**,
**Tooling**, **Archive & retracted**.

Every page is generated at build time from a file that still lives where it
always lived in the repository — `README.md`, `docs/`, `research/`, `AGENTS.md`.
There is no second copy of any fact, so a page cannot disagree with its source:
if it did, one of them would be wrong and the other would be a lie about the
first. The [IA](https://github.com/sehaxe/dormouse/blob/main/docs-site/IA.md)
explains why this project is built that way.

## Where to start

| if you want | go to |
|---|---|
| the current measured state, newest first | [Status](/start-here/status/) |
| the one number that decides whether the model learned language | [Has any checkpoint beaten a 5-gram byte counter?](/archive/five-gram/) |
| to hold the system in your head | [The system map](/architecture/context/) |
| the words, before anything else | [The glossary](/architecture/glossary/) |
| to know what a claim in this repo is allowed to say | [Protocols](/protocols/) |
| why a mechanism is in the codebase at all | [ADR](/adr/) |
| what was claimed and taken back | [Archive & retracted](/archive/) |
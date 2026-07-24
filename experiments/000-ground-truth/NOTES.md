# ground truth (brute force)

## Idea

Not a technique. Renders the raw 2.17M-tri Calamagrostis GCMESH1 community
tile, periodically repeated `tiles × tiles` (its native semantic, 0.52m
period), placed on the shared terrain, with the shared wind model applied per
vertex (weighted by normalized height, so tips sway and roots hold).

This is the **visual reference** every real experiment should A/B against:

    #/ab/000-ground-truth/<your-experiment>?cam=grazing&seed=42

## VRAM budget math

~59MB (32.8MB vertices + 26MB indices) — far over the 25MB budget, which is
fine: `status: reference` experiments are exempt from the perf/VRAM rules.
The HUD bar will show red; that is honest.

## Bake

None — the whole point is rendering the unprocessed source.

## Status

reference

## Findings

Tile count is quadratic: `tiles=3` ≈ 19.5M tris/frame, `tiles=8` ≈ 139M.
Use small tile counts on mid hardware; it only needs to look right up close.

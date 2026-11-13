# VybForge brand

One mascot, one icon, one palette. Everything here is original vector work for
this repo and is covered by the repo `LICENSE`.

## Billet

**Billet** is the mascot: a block of hot iron that got tired of being stock and
started smithing. The visor is a molten **V** — Vyb, struck into the face and
again into the hammer head. The pose is a shop-floor lean: the hammer rests
head-down on the floor, not raised. VybForge drafts desired state; it never
swings that hammer at a host.

- Personality: patient, precise, faintly amused. Reads specs, not instructions
  to execute.
- Never shown mid-swing, never holding a second tool, never wearing a helmet
  over the visor — the visor is the mark.

## Files

| file | what it is |
| --- | --- |
| `mascot.svg` | full character, transparent, 512 × 512 viewBox |
| `icon.svg` | app icon: dark rounded tile, head only, 512 × 512 |
| `icon-mono.svg` | single-ink monogram (outline + V), for stamps, READMEs, embossing |
| `png/mascot-{1024,512,256,128}.png` | raster exports of the mascot |
| `png/icon-{512,256,128,64,48,32,16}.png` | raster exports of the icon |
| `png/favicon.ico` | 16/32/48 multi-size favicon |
| `preview.html` | brand sheet — sizes at 1:1, palette, lockup, rules |

Regenerate the rasters with Inkscape:

```sh
cd assets/brand
for s in 512 256 128 64 48 32 16; do inkscape icon.svg -o png/icon-$s.png -w $s -h $s; done
for s in 1024 512 256 128;    do inkscape mascot.svg -o png/mascot-$s.png -w $s -h $s; done
convert png/icon-16.png png/icon-32.png png/icon-48.png png/favicon.ico
```

## Palette

| hex | name | use |
| --- | --- | --- |
| `#070b11` | forge black | page ground, icon tile base |
| `#141d29` | panel iron | cards, panels |
| `#2a3745` | cold iron | body fill mid-tone |
| `#a9bcd4` | steel | hands, highlights, body text on dark |
| `#ff6a10` | magma | the V, eyes, seams, single accent |
| `#ffe9a6` | hot core | inner gradient stop of the glow, specular dots |

Accent discipline: **magma is the only warm hue.** If two warm hues are on
screen at once, one of them is a bug.

## Sizing rules

- Icon: 16 px minimum. Below 24 px the sparks and cooling seams are detail
  noise — the V and the two eye lamps carry the mark.
- Mascot: 96 px wide minimum. Below that, use `icon.svg`.
- Clear space around either mark: 25 % of its width on all sides.
- Do not recolor the visor, rotate the character, add a second glow hue, or
  place the transparent mascot on a light background without the dark tile.

## Copy

- Name: VybForge (one word, capital V and F). The lockup colors the `Forge`.
- Tagline: *desired state, forged*
- Voice: shop-floor plain. Short sentences, concrete nouns, no exclamation
  marks. Billet states what it will and will not do.

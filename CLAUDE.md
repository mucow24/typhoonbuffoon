# typhoonbuffoon

A twist on the bridge-builder genre: build disaster defences, then run the storm
against them. See `docs/PLAN.md` for architecture and the decisions behind it.

---

## Code testing policy

**This is not advisory. It is the bar for calling anything done.**

- All code, save for the top-most interaction code and other rare exceptions
  (which must be documented clearly in the PR description), must be unit /
  integrated / e2e tested as appropriate.
- These tests shall be written in a TDD / red-first style *as is practical*.
  Don't write harnesses that are larger and more problematic than the bugs they
  attempt to prevent.
- No bullshit tests. "My module isn't loading" is NOT a red test. "My function
  is not returning the expected result" IS. If tests come after implementation,
  use mutation testing to confirm they go red as expected.

### What counts as an exception

Only the outermost layer that cannot be exercised without a browser:

| Exempt | Why |
| --- | --- |
| `src/render/*` | Pixi draw calls. Verified by looking at the screen. |
| `src/ui/*` | DOM panel construction. |
| `src/input/*`, `src/editor/tools.ts` | Pointer and key event plumbing. |
| `src/main.ts`, `src/app.ts` | Composition root and wiring. |

Everything under `src/sim/`, `src/game/`, `src/model/`, `src/world/`,
`src/core/`, `src/scenes/` is testable and must be tested. Geometry and
transform *maths* is testable even when the drawing around it is not:
`src/render/camera.ts` and `src/ui/sliderGeometry.ts` are NOT exempt.

### Why this exists

The first thirteen steps of this project were reported as "verified" on the
strength of scripts that printed numbers I found agreeable. They were written
after the code, exercised only the paths that happened to work, and were never
checked for whether they could fail at all. The renderer had the sky painted
over the entire world and the viewport wrong by 1.5x; the fluid detonated on the
one path the game actually uses. Every one of those was found by a human looking
at the screen, and every one had a passing "verification" behind it.

A test that cannot fail is worse than no test, because it is reported as
confidence.

### Rules that follow from that

1. **Prove the test can fail.** Red-first, or mutate the implementation
   afterwards and watch it go red. State in the PR which method was used.
2. **Test the path the product actually uses.** The fluid was validated on a
   block spawned on a clean grid; the game floods from the field edges. The
   test passed and the feature was broken.
3. **Assert against something external** - a closed-form solution, a
   conservation law, a published result, a hand calculation. Asserting that the
   code does what the code does proves nothing.
4. **Physics gets simulation-level tests**, not just unit tests of the maths.
   See `test/harness/`. A correct kernel and a correct constraint can still
   produce a fluid that explodes when combined.
5. **Look at the screen.** Rendering is exempt from unit tests, not from
   verification. Check it every step, not after thirteen.

---

## Commands

```
npm run dev        # vite dev server
npm test           # vitest, watch
npm run test:run   # vitest, single pass
npm run typecheck  # tsc --noEmit
npm run build      # typecheck + production build
```

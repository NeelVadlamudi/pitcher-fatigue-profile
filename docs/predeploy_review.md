# Pre-Deploy Improvement Review

Reviewed July 27, 2026 against the supplied 675-line checklist, the installed
Streamlit 1.60 runtime, official Streamlit documentation, official Statcast
field documentation, Hugging Face documentation, and the cited public
visualization project.

| Checklist item | Decision | Implementation |
|---|---|---|
| Explicit display names | Applied | Every model feature has an explicit label; unknown fields warn and remain visibly raw. |
| White summary sheet and coach-facing layout | Applied | Feature importance was removed from the PNG and replaced by key findings; the diagnostic remains in the app. |
| Logan Webb narrative | Applied with stricter language | The README avoids unsupported durability categories and describes ASI only on its theoretical scale. |
| GitHub math delimiters | Applied | Display equations use `$$`. |
| Replace `width="stretch"` | Rejected as outdated | Streamlit 1.60 documents `width="stretch"` as current and deprecates or removes `use_container_width`. |
| Empty-data guard | Applied | Fewer than 100 returned pitches produces a clear user-facing stop state. |
| Field-specific coverage | Applied | Velocity, spin, extension, horizontal break, and vertical break coverage are reported. |
| Late-vs-early missingness diagnostic | Applied with a correction | Minimum sample checks use total rows, and missingness is pooled within each window rather than averaging sparse pitch-count rates. |
| ASI component labeling | Applied with a correction | All three components are named. If velocity or late-game coverage is inadequate, the 100-point score is unavailable instead of silently dropping components. |
| Plain-language late-uptick note | Applied | Low-coverage late upticks receive a conditional, non-causal annotation in the static export. |
| Low-R² interpretation | Applied | The app distinguishes pitch-level prediction from the game-level decay curve. |
| Cache freshness | Applied | Provenance contains a UTC cache timestamp and upstream source; the app also offers an explicit refresh control. |
| 150/300-DPI downloads | Applied | The app provides exact 2100×1200 web and 4200×2400 print exports. |
| Credits and palette attribution | Applied | README and PNG credits were added; pitch colors appear only on pitch-type comparisons. |
| Catcher's-perspective movement labels | Not applicable | The shipped movement view is a magnitude-over-pitch-count curve, not an x/y movement plane. |
| Docker port and Hugging Face metadata | Confirmed | Docker exposes port 7860 and README metadata uses `sdk: docker`, `app_port: 7860`. |
| Pre-cache named pitchers | Deferred to deployment operations | Raw Statcast caches are intentionally excluded from version control. The supplied “current Red Sox” list was also stale for 2026; roster-dependent caches must be date-stamped and refreshed at deployment. |
| Push to GitHub and deploy to Hugging Face | External handoff pending | Publishing requires the user's chosen repository/Space and authorization. |
| Resume URL update | External handoff pending | A live URL does not yet exist and is not fabricated. |

## Important source checks

- Streamlit 1.60:
  [`st.dataframe`](https://docs.streamlit.io/develop/api-reference/data/st.dataframe)
  and
  [`st.download_button`](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)
  deprecate `use_container_width` in favor of `width`.
- Streamlit's 2026 release notes document removal of
  `use_container_width` from `st.plotly_chart`.
- Statcast documents `pfx_x` and `pfx_z` in feet and from the catcher's
  perspective; feature deltas are converted to inches before display.
- Hugging Face deprecated the built-in Streamlit SDK on April 30, 2025 and
  directs Streamlit deployments to Docker Spaces.
- PMID 22344048 is Bradbury and Forman's 2012 MLB performance study, not a
  Fleisig biomechanics paper.

## Presentation pass

- Source Sans is the single interface family and is served by Streamlit itself.
- The dashboard follows one bounded content grid with aligned metric cards,
  chart frames, tabs, filters, and downloads.
- Static exports use a compact editorial hierarchy with the pitcher-season as
  the headline and the project name as a small label.
- The README is a case-study landing page; detailed methods remain in `docs/`.
- `docs/github_publishing.md` records the recommended repository name,
  description, topics, social preview, and pre-push checks without inventing a
  repository URL.

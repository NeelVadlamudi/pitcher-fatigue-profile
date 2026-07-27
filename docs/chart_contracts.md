# Chart Contracts

| Surface section | Analytical question | Form | Fields | Supported takeaway | Palette |
|---|---|---|---|---|---|
| Overview | How does fastball velocity change by game pitch count? | Line with interval | Bucket midpoint, equal-game mean, bootstrap bounds, games, pitches, coverage | Shape and supported threshold range | Single blue root, gold reference and threshold band |
| Overview | Which pitch types have the most negative slopes? | Horizontal bar with interval | Pitch type, slope per 10 pitches, bootstrap bounds, games | Relative descriptive degradation | Explicit Nestico-derived pitch-type palette plus interval keylines |
| Pitch quality | How do velocity, spin, or movement deltas change for selected types? | Filtered line with interval | Selected metric and pitch types by bucket | Metric-specific shape with sample context | Blue root, neutral zero |
| Model validation | Do held-out predictions track observed deltas? | Scatter with identity line | Actual delta, predicted delta, game, absolute error | Generalization spread and error | Blue points, neutral identity |
| Model validation | Which variables influence the descriptive model most? | Horizontal bar | Feature, importance, share | Relative model reliance, not causality | Navy root |
| Summary PNG | What should a coach or decision-maker retain? | Two analytical charts plus findings and metrics | Velocity curve, pitch-type slopes, threshold, coverage, held-out MAE, ASI | Fast scan of evidence and guardrails | White background; Red Sox chrome only; pitch colors only on slope bars |

All charts use neutral descriptive titles, visible units, sample context in the
subtitle or hover state, quiet grid lines, and no redundant legends. The final
QA surfaces are the Streamlit app plus 150-DPI and 300-DPI summary exports.
Movement-magnitude curves have no directional x/y movement plane, so catcher's
perspective labels do not apply to the current shipped charts.

## Typography and alignment

- The application uses Streamlit's bundled Source Sans family; no external font
  service or browser request is required.
- The main canvas is capped at 1,180 pixels and follows one left edge from hero
  copy through metrics, tabs, charts, tables, and exports.
- Plotly chart titles use a 19-pixel semibold hierarchy with 13-pixel body text
  and 12-pixel axis text.
- Summary PNGs use Matplotlib's bundled DejaVu Sans fallback so exports render
  consistently in the local environment and Docker image.
- Red Sox colors remain interface accents. Analytical identity is carried by
  the blue trend line, gold reference marks, and pitch-type palette.

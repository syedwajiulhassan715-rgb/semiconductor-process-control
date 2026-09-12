// ===========================================================================
// SECOM process-control model -- Power Query (M)
//
// HOW TO USE
//   1. Power BI Desktop > Home > Transform data > Advanced Editor.
//   2. Create a blank query named  SourceFolder  and paste the first block.
//      Point it at the data/powerbi folder produced by scripts/run_analysis.py.
//   3. Create one blank query per block below, naming it exactly as the
//      comment header states. Names matter -- the DAX in measures.dax refers
//      to these table names.
//   4. Close & Apply, then follow powerbi/README.md for relationships and for
//      marking dim_date as the date table.
//
// WHY TYPES ARE DECLARED EXPLICITLY
//   Every column is typed by hand rather than left to Table.TransformColumnTypes
//   with auto-detection. Two reasons specific to this data:
//     - The source timestamps were day-first (dd/MM/yyyy). They are exported as
//       ISO-8601, but Power Query still applies the *file* locale on import, so
//       an unpinned type turns 07/10/2008 into 10 July. Declaring the type with
//       an invariant culture removes the ambiguity.
//     - Several measure columns are legitimately null (model_score is only
//       populated on the forward test window). Auto-detection samples the first
//       200 rows, sees only nulls, and types the column as text, which silently
//       breaks every downstream aggregation.
// ===========================================================================


// --- Query: SourceFolder ---------------------------------------------------
// Change this one line to relocate the whole model.
"C:\Users\syedw\Downloads\September 2026\semiconductor-process-control\data\powerbi"


// --- Query: fnLoadCsv ------------------------------------------------------
// Shared loader: one place that fixes encoding, delimiter and header handling.
let
    fnLoadCsv = (fileName as text) as table =>
        let
            Source = Csv.Document(
                File.Contents(SourceFolder & "\" & fileName),
                [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]
            ),
            Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true])
        in
            Promoted
in
    fnLoadCsv


// --- Query: dim_date -------------------------------------------------------
let
    Source = fnLoadCsv("dim_date.csv"),
    Typed = Table.TransformColumnTypes(
        Source,
        {
            {"date_key", Int64.Type},
            {"date", type date},
            {"year", Int64.Type},
            {"month", Int64.Type},
            {"month_name", type text},
            {"day_of_month", Int64.Type},
            {"day_name", type text},
            {"iso_year", Int64.Type},
            {"iso_week", Int64.Type},
            {"year_week", type text},
            {"is_weekend", type logical},
            {"has_production", type logical}
        },
        "en-US"
    ),
    // Sort keys so week labels order chronologically rather than alphabetically.
    WithSort = Table.AddColumn(
        Typed, "year_week_sort",
        each [iso_year] * 100 + [iso_week], Int64.Type
    ),
    Keyed = Table.AddKey(WithSort, {"date_key"}, true)
in
    Keyed


// --- Query: dim_sensor -----------------------------------------------------
// Carries all 590 raw sensors, including the 178 excluded by the audit, so the
// report can answer "why is this sensor not on the chart?" without leaving
// Power BI.
let
    Source = fnLoadCsv("dim_sensor.csv"),
    Typed = Table.TransformColumnTypes(
        Source,
        {
            {"sensor_id", type text},
            {"n_present", Int64.Type},
            {"missing_frac", type number},
            {"n_unique", Int64.Type},
            {"mean", type number},
            {"sd", type number},
            {"min", type number},
            {"max", type number},
            {"iqr", type number},
            {"skew", type number},
            {"excess_kurtosis", type number},
            {"robust_outlier_frac", type number},
            {"disposition", type text},
            {"duplicate_of", type text},
            {"retained", type logical},
            {"drift_rho", type number},
            {"drift_q", type number},
            {"cliffs_delta_regime", type number},
            {"q_regime", type number},
            {"cliffs_delta_fail_all", type number},
            {"q_fail_all", type number},
            {"cliffs_delta_fail_stable", type number},
            {"q_fail_stable", type number},
            {"baseline_center", type number},
            {"baseline_sigma", type number},
            {"baseline_lcl", type number},
            {"baseline_ucl", type number},
            {"baseline_n_excluded", Int64.Type},
            {"baseline_median", type number},
            {"baseline_mad", type number},
            {"is_watchlist", type logical},
            {"evidence_tier", type text}
        },
        "en-US"
    ),
    // Readable percentage for report labels; keeps the raw fraction intact.
    WithPct = Table.AddColumn(
        Typed, "missing_pct_label",
        each Number.Round([missing_frac] * 100, 1), type number
    ),
    Keyed = Table.AddKey(WithPct, {"sensor_id"}, true)
in
    Keyed


// --- Query: fact_run -------------------------------------------------------
// Grain: one production run. This is the table every yield measure aggregates.
let
    Source = fnLoadCsv("fact_run.csv"),
    Typed = Table.TransformColumnTypes(
        Source,
        {
            {"run_id", Int64.Type},
            {"timestamp", type datetime},
            {"date_key", Int64.Type},
            {"year_week", type text},
            {"shift", type text},
            {"hour", Int64.Type},
            {"run_index", Int64.Type},
            {"is_fail", type logical},
            {"fail_count", Int64.Type},
            {"minutes_since_prev", type number},
            {"phase", type text},
            {"cusum", type number},
            {"cusum_signal", type logical},
            {"cusum_episode_start", type logical},
            {"model_score", type number}
        },
        "en-US"
    ),
    // Explicit pass counter so yield can be expressed additively rather than
    // as 1 - [Fail Rate], which does not aggregate correctly across a
    // hierarchy with uneven subgroup sizes.
    WithPass = Table.AddColumn(
        Typed, "pass_count", each if [is_fail] then 0 else 1, Int64.Type
    )
in
    WithPass


// --- Query: fact_measurement -----------------------------------------------
// Grain: one watchlist sensor reading on one run (~55k rows).
// robust_z and beyond_limits are pre-computed against FIXED Phase I baselines.
// They are deliberately not recalculated in DAX: a baseline that moves when the
// user changes a slicer is not a baseline.
let
    Source = fnLoadCsv("fact_measurement.csv"),
    Typed = Table.TransformColumnTypes(
        Source,
        {
            {"run_id", Int64.Type},
            {"sensor_id", type text},
            {"value", type number},
            {"baseline_center", type number},
            {"baseline_sigma", type number},
            {"baseline_lcl", type number},
            {"baseline_ucl", type number},
            {"robust_z", type number},
            {"beyond_limits", type logical}
        },
        "en-US"
    ),
    // Sigma-zone banding drives conditional formatting and the Nelson-style
    // zone visuals without a second round trip to Python.
    WithZone = Table.AddColumn(
        Typed, "sigma_zone",
        each let a = Number.Abs([robust_z]) in
            if a = null then "unknown"
            else if a > 3 then "beyond 3 sigma"
            else if a > 2 then "zone A (2-3 sigma)"
            else if a > 1 then "zone B (1-2 sigma)"
            else "zone C (within 1 sigma)",
        type text
    )
in
    WithZone


// --- Query: fact_yield_week ------------------------------------------------
// Pre-computed weekly p-chart. The DAX in measures.dax recomputes these limits
// dynamically so the chart responds to slicers; this table is the Python
// reference to validate the DAX against. If the two disagree, the DAX is wrong.
let
    Source = fnLoadCsv("fact_yield_week.csv"),
    Typed = Table.TransformColumnTypes(
        Source,
        {
            {"year_week", type text},
            {"n", Int64.Type},
            {"defectives", Int64.Type},
            {"p", type number},
            {"center", type number},
            {"lcl", type number},
            {"ucl", type number},
            {"z", type number},
            {"in_baseline", type logical},
            {"above_ucl", type logical},
            {"below_lcl", type logical},
            {"violation", type logical},
            {"naive_ucl", type number},
            {"naive_violation", type logical}
        },
        "en-US"
    )
in
    Typed


// --- Query: fact_gchart ----------------------------------------------------
// Grain: one gap between consecutive failures. Used for the low-rate chart that
// the proportion chart structurally cannot provide.
let
    Source = fnLoadCsv("fact_gchart.csv"),
    Typed = Table.TransformColumnTypes(
        Source,
        {
            {"failure_index", Int64.Type},
            {"run_index", Int64.Type},
            {"runs_since_previous_fail", Int64.Type},
            {"center", type number},
            {"lcl", type number},
            {"ucl", type number},
            {"below_lcl", type logical},
            {"above_ucl", type logical},
            {"violation", type logical}
        },
        "en-US"
    )
in
    Typed

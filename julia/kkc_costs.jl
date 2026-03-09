"""
    KKC Cost Optimizer

Reads a word export CSV from `kkc_builder --export`, computes IME-optimized
costs, and outputs:
  1. `adjusted.csv` — same columns with adjusted cost column
  2. `params.json`  — optimized α/β parameters

# Cost Adjustment Strategy

Sudachi's original costs are tuned for NLP tokenization (fine segmentation).
For IME KKC, we want:

1. **Frequency bias**: common words should have lower cost
2. **Length preference**: longer compounds preferred over fine splits
3. **POS weighting**: 名詞,動詞,形容詞 > 助詞,助動詞 for standalone conversion
4. **Reading ambiguity**: readings with fewer candidates get a bonus
   (less ambiguous → user more likely means exactly that)

The adjusted cost formula:

    C_ime(w) = C_orig(w)
             + δ_freq(w)         # frequency rank adjustment
             + δ_pos(w)          # POS category adjustment
             + δ_ambiguity(w)    # reading ambiguity penalty/bonus
             - δ_length(w)       # length preference boost

Usage:
    julia kkc_costs.jl <input.csv> <output_dir>
"""

using Statistics
using Printf
using Dates

# ─── Configuration ───────────────────────────────────────────────────────────

# POS category cost adjustments (δ_pos)
# Negative = boost (prefer), Positive = penalize
const POS_ADJUSTMENTS = Dict{String, Int}(
    "名詞"   => -200,    # Nouns: strongly prefer
    "動詞"   => -150,    # Verbs: prefer
    "形容詞" => -100,    # Adjectives: prefer
    "副詞"   => -50,     # Adverbs: slight preference
    "連体詞" => -50,     # Pre-noun adjectival: slight preference
    "接続詞" => 0,       # Conjunctions: neutral
    "感動詞" => 0,       # Interjections: neutral
    "助詞"   => 200,     # Particles: penalize (usually not standalone conversion target)
    "助動詞" => 200,     # Auxiliary verbs: penalize
    "接頭辞" => 100,     # Prefixes: penalize
    "接尾辞" => 100,     # Suffixes: penalize
    "記号"   => 300,     # Symbols: strongly penalize
    "補助記号" => 300,   # Supplementary symbols: strongly penalize
    "空白"   => 500,     # Whitespace: heavily penalize
)

# ─── Data Types ──────────────────────────────────────────────────────────────

struct WordEntry
    word_id::UInt32
    reading_hiragana::String
    reading_katakana::String
    surface::String
    cost::Int16
    left_id::Int16
    right_id::Int16
    pos_id::UInt16
    pos_str::String
    char_count::Int
end

# ─── CSV Loading ─────────────────────────────────────────────────────────────

"""
    load_export_csv(path) → Vector{WordEntry}

Load the CSV exported by `kkc_builder --export`.
"""
function load_export_csv(path::String)
    entries = WordEntry[]
    open(path, "r") do io
        header = readline(io)  # skip header

        for line in eachline(io)
            fields = parse_csv_fields(line)
            length(fields) >= 10 || continue

            word_id   = tryparse(UInt32, fields[1])
            reading_h = fields[2]
            reading_k = fields[3]
            surface   = fields[4]
            cost      = tryparse(Int16, fields[5])
            left_id   = tryparse(Int16, fields[6])
            right_id  = tryparse(Int16, fields[7])
            pos_id    = tryparse(UInt16, fields[8])
            pos_str   = fields[9]
            char_count = tryparse(Int, fields[10])

            isnothing(word_id)    && continue
            isnothing(cost)       && continue
            isnothing(left_id)    && continue
            isnothing(right_id)   && continue
            isnothing(pos_id)     && continue
            isnothing(char_count) && continue

            push!(entries, WordEntry(
                word_id, reading_h, reading_k, surface,
                cost, left_id, right_id, pos_id,
                pos_str, char_count,
            ))
        end
    end
    entries
end

"""
    parse_csv_fields(line) → Vector{String}

Parse a CSV line handling quoted fields.
"""
function parse_csv_fields(line::AbstractString)
    fields = String[]
    current = IOBuffer()
    in_quotes = false
    i = 1
    chars = collect(line)
    n = length(chars)

    while i ≤ n
        c = chars[i]
        if in_quotes
            if c == '"'
                if i < n && chars[i+1] == '"'
                    write(current, '"')
                    i += 1
                else
                    in_quotes = false
                end
            else
                write(current, c)
            end
        elseif c == '"'
            in_quotes = true
        elseif c == ','
            push!(fields, String(take!(current)))
        else
            write(current, c)
        end
        i += 1
    end
    push!(fields, String(take!(current)))
    fields
end

# ─── Cost Analysis ───────────────────────────────────────────────────────────

"""
    analyze_costs(entries) → NamedTuple

Compute statistics about the word cost distribution.
"""
function analyze_costs(entries::Vector{WordEntry})
    costs = Float64[e.cost for e in entries]

    by_pos = Dict{String, Vector{Float64}}()
    for e in entries
        # Extract primary POS (first component before '-')
        primary_pos = split(e.pos_str, '-')[1]
        push!(get!(by_pos, primary_pos, Float64[]), Float64(e.cost))
    end

    by_charcount = Dict{Int, Vector{Float64}}()
    for e in entries
        push!(get!(by_charcount, e.char_count, Float64[]), Float64(e.cost))
    end

    # Reading ambiguity: how many entries share each reading
    reading_counts = Dict{String, Int}()
    for e in entries
        reading_counts[e.reading_hiragana] = get(reading_counts, e.reading_hiragana, 0) + 1
    end
    ambiguity_vals = collect(values(reading_counts))

    (
        n           = length(entries),
        μ           = mean(costs),
        σ           = std(costs),
        median      = median(costs),
        p5          = quantile(costs, 0.05),
        p95         = quantile(costs, 0.95),
        by_pos      = by_pos,
        by_charcount = by_charcount,
        μ_ambiguity = mean(Float64.(ambiguity_vals)),
        max_ambiguity = maximum(ambiguity_vals),
        unique_readings = length(reading_counts),
    )
end

"""
    print_analysis(stats)

Print a formatted analysis report.
"""
function print_analysis(stats)
    println("╔═══════════════════════════════════════════════════════════╗")
    println("║           KKC Dictionary Cost Analysis                   ║")
    println("╠═══════════════════════════════════════════════════════════╣")
    @printf("║ Total entries:     %8d                              ║\n", stats.n)
    @printf("║ Unique readings:   %8d                              ║\n", stats.unique_readings)
    @printf("║ Mean cost:         %8.1f                              ║\n", stats.μ)
    @printf("║ Std dev:           %8.1f                              ║\n", stats.σ)
    @printf("║ Median cost:       %8.1f                              ║\n", stats.median)
    @printf("║ P5/P95:            %8.1f / %-8.1f                   ║\n", stats.p5, stats.p95)
    @printf("║ Mean ambiguity:    %8.1f entries/reading              ║\n", stats.μ_ambiguity)
    @printf("║ Max ambiguity:     %8d entries/reading              ║\n", stats.max_ambiguity)
    println("╚═══════════════════════════════════════════════════════════╝")
    println()

    # Cost by POS
    println("┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
    println("│ POS         │ Count    │ Mean     │ Median   │ Std      │")
    println("├─────────────┼──────────┼──────────┼──────────┼──────────┤")
    for (pos, costs) in sort(collect(stats.by_pos), by=x->-length(x[2]))
        length(costs) >= 5 || continue
        @printf("│ %-11s │ %8d │ %8.1f │ %8.1f │ %8.1f │\n",
            first(pos, 11), length(costs), mean(costs), median(costs), std(costs))
    end
    println("└─────────────┴──────────┴──────────┴──────────┴──────────┘")
    println()

    # Cost by character count
    println("┌──────┬──────────┬──────────┬──────────┐")
    println("│ Len  │ Count    │ Mean     │ Median   │")
    println("├──────┼──────────┼──────────┼──────────┤")
    for len in sort(collect(keys(stats.by_charcount)))
        costs = stats.by_charcount[len]
        length(costs) >= 5 || continue
        @printf("│ %4d │ %8d │ %8.1f │ %8.1f │\n",
            len, length(costs), mean(costs), median(costs))
    end
    println("└──────┴──────────┴──────────┴──────────┘")
end

# ─── Cost Optimization ───────────────────────────────────────────────────────

"""
    compute_adjusted_costs(entries) → (adjusted_entries, params)

Apply IME cost adjustments to each entry:

1. **δ_freq**: Rank-based adjustment within each reading group.
   Within entries sharing the same reading, sort by original cost;
   the top-ranked entry gets a bonus, lower-ranked get penalties.

2. **δ_pos**: POS category adjustment from POS_ADJUSTMENTS table.

3. **δ_ambiguity**: Entries with unique/rare readings get a bonus
   (high confidence in conversion).

4. **δ_length**: Longer surface forms get boosted to prefer compounds.

5. **α/β**: Derive compound boost and identity penalty from the data.
"""
function compute_adjusted_costs(entries::Vector{WordEntry})
    # ── Reading ambiguity map ──
    reading_counts = Dict{String, Int}()
    for e in entries
        reading_counts[e.reading_hiragana] = get(reading_counts, e.reading_hiragana, 0) + 1
    end
    μ_ambiguity = mean(Float64.(values(reading_counts)))

    # ── Group by reading for frequency ranking ──
    reading_groups = Dict{String, Vector{Int}}()
    for (i, e) in enumerate(entries)
        push!(get!(reading_groups, e.reading_hiragana, Int[]), i)
    end

    # ── Compute adjustments ──
    adjusted_costs = Vector{Int16}(undef, length(entries))

    for (reading, indices) in reading_groups
        # Sort by original cost within this reading group (lower = better)
        sorted_indices = sort(indices, by=i -> entries[i].cost)
        n_group = length(sorted_indices)

        for (rank, idx) in enumerate(sorted_indices)
            e = entries[idx]
            cost = Int(e.cost)

            # δ_freq: rank-based adjustment within reading group
            # Top candidate gets bonus, others get progressive penalty
            if n_group > 1
                # Normalize rank to [0, 1], where 0 = best
                rank_ratio = (rank - 1) / (n_group - 1)
                # Top entry: -300 bonus, worst: +200 penalty
                δ_freq = round(Int, -300 + 500 * rank_ratio)
            else
                δ_freq = -200  # unique reading → bonus
            end

            # δ_pos: POS category adjustment
            primary_pos = split(e.pos_str, '-')[1]
            δ_pos = get(POS_ADJUSTMENTS, primary_pos, 0)

            # δ_ambiguity: reading ambiguity adjustment
            ambiguity = reading_counts[e.reading_hiragana]
            if ambiguity ≤ 3
                δ_ambi = -100   # low ambiguity = confident conversion
            elseif ambiguity ≤ 10
                δ_ambi = 0
            else
                δ_ambi = round(Int, min(200, 10 * log2(ambiguity)))
            end

            # δ_length: surface length preference
            # Longer entries get boosted (prefer compounds)
            δ_len = -50 * max(0, e.char_count - 1)

            # Apply all adjustments
            adjusted = cost + δ_freq + δ_pos + δ_ambi + δ_len

            # Clamp to i16 range
            adjusted_costs[idx] = Int16(clamp(adjusted, typemin(Int16), typemax(Int16)))
        end
    end

    # ── Derive α and β ──
    # α: per-extra-character compound boost
    # Compare cost savings between 1-char and 2-char entries
    costs_1 = Float64[entries[i].cost for i in 1:length(entries) if entries[i].char_count == 1]
    costs_2 = Float64[entries[i].cost for i in 1:length(entries) if entries[i].char_count == 2]

    α = if !isempty(costs_1) && !isempty(costs_2)
        cost_per_char_savings = median(costs_1) - median(costs_2) / 2
        round(Int, clamp(abs(cost_per_char_savings) * 0.7, 200, 1500))
    else
        500
    end

    # β: identity penalty
    # Set above 95th percentile of single-char costs
    β = if !isempty(costs_1)
        round(Int, clamp(quantile(costs_1, 0.95) * 1.5, 8000, 20000))
    else
        12000
    end

    (adjusted_costs, (α=Int16(α), β=Int16(β)))
end

# ─── Output ──────────────────────────────────────────────────────────────────

"""
    write_adjusted_csv(path, entries, adjusted_costs)

Write the adjusted CSV with the same format as input but updated costs.
"""
function write_adjusted_csv(path::String, entries::Vector{WordEntry}, adjusted_costs::Vector{Int16})
    open(path, "w") do io
        println(io, "word_id,reading_hiragana,reading_katakana,surface,cost,left_id,right_id,pos_id,pos_str,char_count")
        for (i, e) in enumerate(entries)
            surface = csv_escape(e.surface)
            reading_h = csv_escape(e.reading_hiragana)
            reading_k = csv_escape(e.reading_katakana)
            pos = csv_escape(e.pos_str)
            println(io, "$(e.word_id),$reading_h,$reading_k,$surface,$(adjusted_costs[i]),$(e.left_id),$(e.right_id),$(e.pos_id),$pos,$(e.char_count)")
        end
    end
end

function csv_escape(s::AbstractString)
    if occursin(r"[,\"\n]", s)
        "\"" * replace(s, "\"" => "\"\"") * "\""
    else
        s
    end
end

"""
    write_params_json(path, params)

Write params.json with α and β values.
"""
function write_params_json(path::String, params)
    open(path, "w") do io
        println(io, """{
  "alpha": $(params.α),
  "beta": $(params.β),
  "generated": "$(Dates.now())",
  "description": "IME cost parameters derived by Julia KKC optimizer"
}""")
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) < 2
        println(stderr, "Usage: julia kkc_costs.jl <input.csv> <output_dir>")
        println(stderr, "")
        println(stderr, "Reads word export CSV from kkc_builder --export,")
        println(stderr, "computes IME-optimized costs, and outputs:")
        println(stderr, "  <output_dir>/adjusted.csv  — entries with adjusted costs")
        println(stderr, "  <output_dir>/params.json   — α/β parameters")
        exit(1)
    end

    input_csv  = ARGS[1]
    output_dir = ARGS[2]

    mkpath(output_dir)

    # Load
    println(stderr, "Loading $input_csv...")
    entries = load_export_csv(input_csv)
    println(stderr, "  Loaded $(length(entries)) entries")

    # Analyze
    stats = analyze_costs(entries)
    print_analysis(stats)

    # Optimize
    println(stderr, "\nComputing IME-adjusted costs...")
    adjusted_costs, params = compute_adjusted_costs(entries)

    # Report cost changes
    orig_costs = Float64[e.cost for e in entries]
    adj_costs  = Float64.(adjusted_costs)
    Δ = adj_costs .- orig_costs
    println(stderr, @sprintf("  Mean cost change:   %+.1f", mean(Δ)))
    println(stderr, @sprintf("  Median cost change: %+.1f", median(Δ)))
    println(stderr, @sprintf("  Range of changes:   [%+.0f, %+.0f]", minimum(Δ), maximum(Δ)))

    # Output
    adjusted_path = joinpath(output_dir, "adjusted.csv")
    params_path   = joinpath(output_dir, "params.json")

    println(stderr, "\nWriting $adjusted_path...")
    write_adjusted_csv(adjusted_path, entries, adjusted_costs)

    println(stderr, "Writing $params_path...")
    write_params_json(params_path, params)

    # Summary
    println()
    println("═══════════════════════════════════════════════════════")
    println("  KKC Cost Optimization Complete")
    println("═══════════════════════════════════════════════════════")
    println("  α (IME_COMPOUND_BOOST) = $(params.α)")
    println("  β (IME_IDENTITY_COST)  = $(params.β)")
    println("  Entries processed:       $(length(entries))")
    println("  Output: $adjusted_path")
    println("           $params_path")
    println("═══════════════════════════════════════════════════════")
    println()
    println("Next step:")
    println("  kkc_builder --build $adjusted_path $params_path output.kkc")
end

main()

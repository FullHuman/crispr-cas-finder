fn check_short_crispr(dr: &str, spacer: &str) -> (usize, usize, f64, f64) {
    let a = dr.as_bytes();
    let b = spacer.as_bytes();
    let m = a.len();
    let n = b.len();

    let match_sc: f64 = 5.0;
    let mismatch_sc: f64 = -4.0;
    let gap_open: f64 = 10.0;
    let gap_extend: f64 = 0.5;

    let inf = f64::NEG_INFINITY;

    let mut mm = vec![vec![inf; n + 1]; m + 1];
    let mut ga = vec![vec![inf; n + 1]; m + 1];
    let mut gb = vec![vec![inf; n + 1]; m + 1];

    mm[0][0] = 0.0;
    for cell in ga[0].iter_mut().skip(1) {
        *cell = 0.0;
    }
    for row in gb.iter_mut().skip(1) {
        row[0] = 0.0;
    }

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] {
                match_sc
            } else {
                mismatch_sc
            };
            mm[i][j] = s + mm[i - 1][j - 1].max(ga[i - 1][j - 1]).max(gb[i - 1][j - 1]);
            ga[i][j] = (mm[i][j - 1] - gap_open - gap_extend)
                .max(ga[i][j - 1] - gap_extend)
                .max(gb[i][j - 1] - gap_open - gap_extend);
            gb[i][j] = (mm[i - 1][j] - gap_open - gap_extend)
                .max(ga[i - 1][j] - gap_open - gap_extend)
                .max(gb[i - 1][j] - gap_extend);
        }
    }

    let mut best_i = m;
    let mut best_j = n;
    let mut best_score = mm[m][n].max(ga[m][n]).max(gb[m][n]);
    for j in 1..n {
        let sc = mm[m][j].max(ga[m][j]).max(gb[m][j]);
        if sc > best_score {
            best_score = sc;
            best_i = m;
            best_j = j;
        }
    }
    for i in 1..m {
        let sc = mm[i][n].max(ga[i][n]).max(gb[i][n]);
        if sc > best_score {
            best_score = sc;
            best_i = i;
            best_j = n;
        }
    }

    let trailing_a_gaps = n - best_j;
    let trailing_b_gaps = m - best_i;

    #[derive(Clone, Copy)]
    enum St {
        Mm,
        Ga,
        Gb,
    }
    let mut state = St::Mm;
    {
        let mut bv = mm[best_i][best_j];
        if ga[best_i][best_j] > bv {
            bv = ga[best_i][best_j];
            state = St::Ga;
        }
        if gb[best_i][best_j] > bv {
            state = St::Gb;
        }
    }

    let mut i = best_i;
    let mut j = best_j;
    let mut gaps = trailing_a_gaps + trailing_b_gaps;
    let mut total = gaps;
    let ep = 1e-6;

    while i > 0 || j > 0 {
        match state {
            St::Mm => {
                if i == 0 || j == 0 {
                    break;
                }
                total += 1;
                let s = if a[i - 1] == b[j - 1] {
                    match_sc
                } else {
                    mismatch_sc
                };
                let prev = mm[i][j] - s;
                if (prev - mm[i - 1][j - 1]).abs() < ep {
                    state = St::Mm;
                } else if (prev - ga[i - 1][j - 1]).abs() < ep {
                    state = St::Ga;
                } else {
                    state = St::Gb;
                }
                i -= 1;
                j -= 1;
            }
            St::Ga => {
                if j == 0 {
                    break;
                }
                total += 1;
                gaps += 1;
                let cur = ga[i][j];
                if i == 0 {
                    j -= 1;
                    continue;
                }
                if (cur - (ga[i][j - 1] - gap_extend)).abs() < ep {
                    state = St::Ga;
                } else if (cur - (mm[i][j - 1] - gap_open - gap_extend)).abs() < ep {
                    state = St::Mm;
                } else {
                    state = St::Gb;
                }
                j -= 1;
            }
            St::Gb => {
                if i == 0 {
                    break;
                }
                total += 1;
                gaps += 1;
                let cur = gb[i][j];
                if j == 0 {
                    i -= 1;
                    continue;
                }
                if (cur - (gb[i - 1][j] - gap_extend)).abs() < ep {
                    state = St::Gb;
                } else if (cur - (mm[i - 1][j] - gap_open - gap_extend)).abs() < ep {
                    state = St::Mm;
                } else {
                    state = St::Ga;
                }
                i -= 1;
            }
        }
    }
    gaps += i + j;
    total += i + j;

    (total, gaps, gaps as f64 / total as f64, best_score)
}

fn main() {
    let dr = "ATGCCTGATGCGACGCTATGCGCGTCTTATCAGGCCTACGG";
    let sp = "TTTATGGGCGAAGTGTAGACCGGATAAGGCGTTCACGCCGCATCCGGCAGTCGTGCGCC";

    let (total, gaps, gap_frac, score) = check_short_crispr(dr, sp);
    println!("End-gap-free NW:");
    println!(
        "  align_len={}, gaps={}, gap%={:.1}%, score={:.1}",
        total,
        gaps,
        gap_frac * 100.0,
        score
    );
    println!("  EMBOSS: align_len=64, gaps=28, gap%=43.8%, score=45.0");
    println!("  accept={} (want false for 377333)", gap_frac > 0.5);
}

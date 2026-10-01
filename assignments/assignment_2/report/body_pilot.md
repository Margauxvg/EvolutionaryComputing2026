Both bodies pass. The pipeline is ready for the comparison:

|  | John Set gecko | spider\_8 |
| :---- | :---- | :---- |
| Hinges (outputs) | 6 | 8 |
| Controller inputs | 12 | 14 |
| Genotype length | 108 | 132 |
| Seconds per evaluation | 0.212 | 0.264 |
| Deterministic | yes | yes |

These numbers are already Methods material for the body-selection paragraph. The gecko is about 20% cheaper per evaluation and has 24 fewer weights. Whether that outweighs how well each body learns is what the pilot will show.

###  **Results at equal budget (1,230 evaluations per run)**

|  | Seed 1 | Seed 2 | Seed 3 | Mean ± sd | Median | Wall time per run |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- |
| **gecko** (6 hinges, 108 weights) | 1.29 | 1.04 | 0.96 | **1.10 ± 0.17** | 1.04 | \~300 s |
| **spider\_8** (8 hinges, 132 weights) | 1.18 | 0.46 | 1.19 | **0.94 ± 0.42** | 1.18 | \~410 s |

**What this shows**

**spider\_8 has the better mean, but only because of one seed.** Seed 2 found a good gait between generations 10 and 30 (1.56 → 0.46). The other two got stuck around 1.18–1.19, with long flat stretches: seed 3 made no progress from generation 20 to 30, and seed 1 none from 10 to 20 or from 30 to 40\. By the median, the gecko is better.

**The gecko is much more consistent.** All three seeds improved steadily, and the spread is less than half of spider\_8's (sd 0.17 vs. 0.42). With spider\_8, the outcome of a run seems to depend mostly on whether it happens to find a gait, which is closer to a lottery. That luck would dominate the spread between seeds, so a real difference between fixed and adaptive σ would be hard to detect and even harder to explain. The gecko's steady curves give the σ variants a clean background to differ against.

**The gecko is cheaper:** about 27% less wall time per run, and 24 fewer weights. With 20 seeds × 3 variants, that saves hours in the final week.

**Both still have room to learn.** Neither reached the target or a clear plateau at generation 40, so the generation count still has to come from a longer pilot on the chosen body.

**Body choice: the gecko**

**fewer hinges means fewer controller inputs and outputs, so fewer weights to evolve (108 vs. 132), so a smaller search space.** There are 4 reasons why to choose for the gecko rather than the spider\_8.

1. **Lower spread between seeds (sd 0.17 vs. 0.42 m).** This is the strongest argument. Your research question compares two mutation schemes across seeds, so the spread between seeds is the noise any difference must stand out against. With spider\_8, the outcome mostly depended on whether a run happened to find a gait: one seed reached 0.46 m, the other two got stuck around 1.18 m. That variance would swamp a σ effect and be hard to explain.  
2. **Steady improvement in every run.** All three gecko seeds improved gradually and none reached the target, so there's room for the variants to differ. That's exactly the kind of curve where *convergence speed*, half of your research question, can actually be measured.  
3. **Fewer dimensions.** The 1/5 rule's behaviour depends on the number of dimensions being searched; Rechenberg's analysis is explicitly dimension-dependent. A smaller genotype gives the 1/5 rule a fair chance within the budget. Th is not just for convenienve but also rather a methodological reason.  
4. **27% less compute per run,** which buys more seeds or generations in the final experiment. Probably seeds as the experiments stops after plateauing. 

Reasons to choose for spider\_8: it had the better *mean*. 

**Overall:** choose the gecko for consistency and interpretability and not just performance. 


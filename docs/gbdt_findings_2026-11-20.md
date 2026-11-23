IC selector: findings from a GBDT test
Nov 20, 2026
Your selector finds junk columns far better than chance, but as a pre-fit screen for a gradient-boosted tree model it cost accuracy. The main cause is the cutoff: it keeps a column only when the column is proven useful, while a screen should drop a column only when it is proven useless. Below are the tests, the four ways it went wrong, and five changes that would point it at this job.
What we ran
We used `nonlinear_corr` and `recursive_ic_score` inside ChimeraBoost, a pure-Python gradient-boosted tree library with CatBoost-style symmetric trees. The selector ran on the training rows only, before any tree was grown. Every column it flagged was blocked from every split for the whole fit, along with any cross feature built from it.
The data was the 59 Grinsztajn benchmark datasets plus 14 datasets with high-cardinality categoricals, 3 seeds each. Each ran at full size and again with only 25% of the training rows. We scored test RMSE for regression and Brier score for classification.
Three rules were compared against the unmasked model:
Conservative: drop a column only if neither marginal test finds anything (ic and xi, Bonferroni at 0.05) and `recursive_ic_score(alpha=0.05)` never picks it.
Aggressive: drop every column the recursive pass did not pick.
Random control: drop as many columns as the conservative rule, chosen at random.
For multiclass targets we took the union of one-vs-rest recursive passes, since the recursive pass needs a numeric target.
Results
Both rules made the model worse on the Grinsztajn datasets, and both beat the random control by a wide margin. Each cell counts datasets where masking made the test score better vs worse. Ties are left out; most are datasets where nothing was dropped. The p-values are two-sided sign tests.
Rule	Grinsztajn, full rows	Grinsztajn, 25% rows	High-card, full rows	High-card, 25% rows
Conservative	7 vs 18 (p = 0.04)	10 vs 26 (p = 0.01)	4 vs 6	7 vs 6
Aggressive	11 vs 27 (p = 0.01)	12 vs 39 (p < 0.001)	5 vs 8	5 vs 9
Random control	1 vs 23	3 vs 34	1 vs 9	2 vs 11
Set directly against the random control, the conservative rule wins 19 vs 5 at full size and 30 vs 7 at 25% rows. So the scores do separate junk from signal well. The model simply loses more from the real columns that get cut than it gains from the junk that goes.
Two things it already does well are worth keeping. Being rank-based, it shrugs off outliers and skew. Thanks to the ordered encoding, an ID-like categorical scores near zero, which is the same trick ChimeraBoost relies on.
Where it goes wrong
1. The cutoff demands proof of usefulness
Bonferroni at 0.05 across every column keeps a column only on strong evidence. For a screen, the costs run the other way: a kept junk column costs a tree model almost nothing, while a dropped real column is gone for good. On 1,000-column synthetic data with 20 real columns, the conservative rule kept only 8 to 15 columns in all, so it dropped at least 5 of the real ones.
The effect grows as data shrinks. At 25% rows, the aggressive rule flagged a median 29% of each Grinsztajn dataset's columns, against 7% at full size. On pol (classification, 25% rows) the conservative rule cut about ten weak columns and Brier got 32% worse.
2. Pairs that only matter together go unseen
On nyc-taxi, pickup minute and dropoff minute each look like noise on their own, but their difference is trip duration. The conservative rule dropped both, and RMSE got 9 to 10% worse. The interaction rescue only stratifies on columns already picked, so when neither partner is picked the pair stays invisible. SGEMM, a GPU kernel timing table built mostly from interactions, came out 26% worse under the aggressive rule at 25% rows.
3. Dropping redundant columns hurts tree models
The recursive pass drops columns that add little beyond the ones already picked. That suits a short list for explanation or a linear model. Tree models gain from correlated copies, which carry different noise and different split points. On Bioresponse (419 molecular descriptors) the aggressive rule kept about 17 columns, and Brier got 12% worse.
4. Row-order and ID columns score as signal
On eye_movements the selector kept `lineNo`, which has a different value on every row, and `assgNo`, an assignment number. It dropped most of the actual eye-tracking measures. xi and binned MI pick up runs of the same label in row order. The model's Brier improved 5.5% there, but only because it leaned harder on that leak.
Where a screen pays
On data that is mostly junk, a perfect screen makes the model both faster and more accurate, so the opportunity is real. Your selector captures most of the speed but loses accuracy to the strict cutoff. The test used 1,000 uniform columns, of which 20 drive the target through Friedman-style terms, one product interaction, and decaying linear, step and sine terms. Times include the selector; 3 seeds each.
Data	Default fit	Your selector (conservative): time, error	Perfect screen: time, error
Regression, 5k rows	8.1 s	0.27x, 4.1% worse	0.20x, 6.1% better
Regression, 20k rows	16.2 s	0.31x, 2.6% worse	0.18x, 6.4% better
Binary, 5k rows	3.9 s	0.48x, flat	0.25x, 3.4% better
Binary, 20k rows	8.4 s	0.46x, flat	0.29x, 1.7% better
Real wide datasets behave differently because most of their columns carry some signal. On the five widest in our benchmarks (79 to 419 columns), the selector cost about as much time as it saved. The best case was Allstate, at 0.57x the time and 3.6% worse. On topo_2_1 (255 columns, 6,663 rows) the selector alone took 9 s against a 2 s fit.
Suggested directions
Add a screening mode that guards against false drops. Use an equivalence test in place of the significance test: drop a column only when the upper confidence bound of its ic sits below a small floor, such as 0.02. At small n the wide intervals keep anything uncertain, and at large n true junk still goes. A cruder version would likely do most of the job: keep anything with an uncorrected p below 0.2 on either test.
Rescue pairs among the leftovers. Before dropping a column, score it inside bins of the other would-be-dropped columns, reusing the stratified tables the recursive pass already builds. Capping this at the K leftovers with the most marginal or variance evidence keeps the cost to K coarse tables per column. It would have kept the nyc-taxi minute pair.
Report relevance and novelty separately. One score would say whether a column carries any signal, alone or in a pair. Another would say what it adds beyond the columns already picked. A screen for tree models should use relevance only, while the recursive ranking stays the right tool for explanation and linear models.
Flag identifier and row-order columns. A column that is nearly unique per row, or tracks row position closely, could get a warning field in place of a high score. eye_movements would then show its leak openly.
Add a fast path and native multiclass. Accepting a pre-binned integer matrix would skip the polars conversion and the per-column sort, and most tree libraries already hold one. The topo_2_1 run is worth profiling. A multiclass target currently needs one recursive pass per class.
We can rerun the same tests on a screening-mode version; the full set takes about 30 minutes.
# Silent Cart: speaker notes (practice script)

Total: 1339 words, about 10 minutes at 130 words per minute. With pauses and slide changes this lands around 11-13 minutes, leaving time for questions in a 15-minute slot.

Tips: pause after each headline number; point at the chart you are describing; slide 4 (two churns) and slide 11 (money) are the moments to slow down.

## Slide 1: Most of our churn has already happened. Here is how to stop the rest.
*~0.7 min · 93 words*

Good morning. FreshBasket asked a simple question: which loyalty members are about to stop shopping, and what should we do about it? Today they send the same offers to everyone. In the next 12 minutes I'll show four things: that the churn number management sees mixes two very different problems; the warning signs that appear about two months before a member leaves; a model that catches about three in four at-risk members on a quarter it never saw; and a targeted retention playbook that turns today's loss-making blanket coupon into a positive return.

## Slide 2: The answer in one slide
*~0.9 min · 114 words*

Here's the whole story on one slide. First, 80% of members labelled as churned had already gone quiet for three months or more before the label window started; that is win-back, not prediction. Second, among members still shopping, the model catches 76% of the ones who leave, with 70% precision, against a base rate under 10%. Third, support experience is a strong lever: members with three or more tickets churn at nearly ten times the rate of members with none. And fourth, targeting changes the economics: the playbook returns about +$3.6k on this cohort where the blanket coupon loses $15.4k. My recommendation is at the bottom; I'll come back to it at the end.

## Slide 3: We fixed the data before trusting it, and found the label leaks
*~0.9 min · 119 words*

Before any modelling I audited the three tables; all 28 checks are logged in a CSV. Six findings matter. The real header is on row two. There are duplicate rows, including two pairs where the same customer-month appears with positive and negative spend. 55 spend values are corrupted; because spend equals transactions times basket for 99.8% of rows, I could repair them exactly instead of deleting months. Missing months are data loss, not inactivity. The most important finding is on the right: three columns in the label table are computed using April to June, the very period we predict. Using them would quietly inflate performance, so they're excluded, and a unit test enforces that features never see the future.

## Slide 4: Two different churn problems hide inside one number
*~0.8 min · 103 words*

This chart changed how I framed the whole project. Churn probability is a cliff in recency: a member who bought last month churns at about 3%; miss one month and it's 38%; miss two and it's 81%; beyond that it's essentially 100%. So the official churn rate of 34% mixes 653 members who were already gone with 1,715 active members who churn at under 10%. That matters for management, because the KPI overstates what retention can influence. It matters for data science too: any model looks brilliant on the lapsed group. So from here on I report every result for active members separately.

## Slide 5: Churn has a fingerprint, and a two-month window to act
*~0.7 min · 93 words*

Here I lined up every active member on their last purchase before April and looked backwards. Retained members, in blue, are flat. Churners, in orange, show the same sequence every time: transactions, app sessions and email opens fade over roughly two months, while support tickets spike. That gives us a concrete intervention window, and simple triggers a CRM team can use without any model: a 40% drop in spend, app use falling below one session every two months, or a formal complaint. All of these are statistically significant after correcting for multiple tests.

## Slide 6: Support experience and engagement matter. Tier does not.
*~0.7 min · 89 words*

The brief asked whether tier, marketing engagement or support experience affect churn. The answer is yes, yes, and surprisingly no. Support friction is the clearest operational lever: three or more tickets takes churn from 4% to 37%. Engagement, meaning app sessions and email opens, sits among the top three SHAP drivers in both the logistic and the gradient-boosting model. And membership tier makes no difference: Platinum members leave at the same rate as Silver. The current perks are not buying loyalty, which I'll come back to in the recommendations.

## Slide 7: Validated the way it will be used: train on the past, test on an unseen quarter
*~0.8 min · 101 words*

Random splits flatter churn models because the future leaks into training. Instead I replayed the churn definition at earlier cut-offs: July and October 2023 for training, January 2024 for model selection, calibration and the threshold, and the official April to June label only once, at the end. Features only look at the six months before each cut-off. Class imbalance is handled with weights, then the scores are calibrated so a 30% really means 30%; that matters for the money maths. Everything is tracked in MLflow, and the feature pipeline also exists in PySpark, verified identical, so it can run on Databricks.

## Slide 8: On Active members, machine learning roughly doubles what a recency rule can do
*~1.0 min · 135 words*

On active members, the recency rule reaches a PR-AUC of 0.46. All three machine-learning models roughly double it, to around 0.8, on a quarter they never saw. Logistic regression won on validation; gradient boosting scores a touch higher on test, but the confidence intervals overlap almost completely, so I deploy the simpler, transparent model and keep boosting as a challenger. At the tuned threshold we flag 179 active members and three out of four real churners are among them. The calibration plot, bottom right, shows the probabilities can be trusted as real odds. I also trained a GRU network on the raw monthly sequences: it scores slightly higher and wins in 97% of bootstrap resamples, so the plan is to shadow-score it for a quarter and promote it if the gain holds: champion and challenger.

## Slide 9: Every member gets a score and three reasons the CRM team can act on
*~1.1 min · 137 words*

A score on its own doesn't get used. On the left is the SHAP summary for active members: low recent purchasing, low email and app engagement, and recent support tickets push risk up. On the right is what the store or CRM team actually receives: each member's churn probability plus three plain-English reasons, drawn from that member's own SHAP values and grouped so the three reasons are different. That's the bridge from a model to an action. One caution: SHAP explains the model, not causation; that's why the next step is an experiment, not a guess. And the last mile is automated too: for the priority members an LLM on Groq drafts the actual message or call script from these same three reasons, under guardrails that block invented promo codes, false apologies and any mention of tracking.

## Slide 10: Behaviour predicts churn. Demographics alone are no better than chance.
*~0.8 min · 99 words*

The brief asked whether behavioural, engagement and support features beat demographics alone. The ablation says clearly yes. With only demographics and tier, the model is at the base rate: no better than chance. Adding purchase behaviour jumps it to 0.66, engagement adds another 0.11, and support pushes it to about 0.8. Even with recency removed the model stays near 0.8, so it is reading genuine early-warning signs. By segment, Austin and Columbus run highest and members who joined in 2024 lowest, but these gaps are small next to behaviour, so segments decide which team acts, not who gets targeted.

## Slide 11: Targeting turns retention from a cost into a return
*~0.9 min · 120 words*

Now the money. Assumptions are explicit: a ten-dollar coupon cuts a member's churn probability by 15%, a fifteen-dollar service call by 25%, and a member's value is six months of their spend at a 25% margin. The blanket coupon, today's approach, costs about $24k on this cohort and loses money, because most recipients were staying anyway or are already gone. The playbook gives each member at most one action and only when it pays back: about 43 churners prevented for $3.9k, a +$3.6k return. The brief's support-outreach idea works, but only when targeted: calling everyone with tickets loses money, while calling the model-flagged ones makes money. The sensitivity chart shows targeted actions break even at roughly half the assumed effect.

## Slide 12: Recommendations: a 90-day plan
*~1.0 min · 136 words*

To close, six actions for the next 90 days. Split the churn KPI so management sees what retention can actually influence, and add a 60-day no-purchase trigger. Run the scoring and next-best-action list every month. Treat support as a retention lever and close the loop inside the two-month window. Use free nudges before paid offers. Revisit tier perks, since Platinum isn't buying loyalty. And prove the uplift with a hold-out control group before scaling, because the economics rest on assumptions until we measure them; about 730 flagged members are enough to detect a 15% uplift. Everything I've shown regenerates from one command, the feature pipeline is ported to Spark for Databricks, and there is a Dash retention console where the CRM team can explore members and test their own assumptions. Thank you; happy to take questions.

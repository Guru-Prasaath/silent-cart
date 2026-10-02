# Silent Cart: speaker notes (practice script)

**Total speaking time: about 11 minutes** at 130 words per minute. With pauses and slide changes this is 11-13 minutes, which fits the 10-15 minute slot.

**How to use this script**
- **Main point** is the one sentence the panel must remember from the slide. If you forget everything else, say this.
- **Say** lists short ideas, not a speech to memorise. Say them in your own words.
- **Point at** tells you where to look or point on the slide.
- **Next slide** is the bridge sentence that leads into the next slide.
- Slow down on **slide 4** (two kinds of churn) and **slide 11** (the money). Those are the moments the panel remembers.
- Open https://silent-cart.onrender.com/ a few minutes before presenting so the live demo is awake.

---

## Slide 1: Most of our churn has already happened. Here is how to stop the rest.

*About 52 seconds*

**Main point:** Most of the "churn" has already happened. I will show how to stop the rest.

**Say:**

- Good morning. I am Guru Prasaath.
- FreshBasket asked two questions: which loyalty members will stop shopping, and what should we do about it?
- Today they send the same offer to everyone. That wastes money.
- In the next 12 minutes I will show four things:
1. The churn number hides two very different problems.
2. Members show warning signs about 2 months before they leave.
3. A model that catches 3 out of 4 leavers, on data it never saw.
4. A targeted plan that makes money instead of losing it.

**Next slide:** Let me start with the answer.

---

## Slide 2: The answer in one slide

*About 45 seconds*

**Main point:** Four numbers tell the whole story.

**Say:**

- 80%: most members labelled "churned" had already stopped buying BEFORE April. They need a win-back offer, not a prediction.
- 76%: among members who are still shopping, the model finds 76% of the ones who leave. When it flags someone, it is right 70% of the time. Normally only about 1 in 10 leave.
- 37%: members with 3 or more support tickets leave at 37%, compared with 4% for members with none. Service problems drive churn.
- $3.6k: my targeted plan earns $3.6k. Today's coupon-for-everyone loses $15.4k.

**Point at:** the dark box at the bottom. These are my 3 recommendations. I will come back to them at the end.

**Next slide:** First, I checked whether the data could be trusted.

---

## Slide 3: We fixed the data before trusting it, and found the label leaks

*About 62 seconds*

**Main point:** I cleaned the data first, and I found 3 columns that "cheat".

**Say:**

- I ran 28 data checks. Every check is logged in a file.
- Problems I fixed: a hidden header row, duplicate rows, 55 broken spend values, impossible zeros and missing months.
- Broken spend: "spend = transactions x basket size" is true for 99.8% of rows. So I could rebuild the wrong values exactly, without deleting any data.
- Most important: 3 columns in the label table use April-June data. That is the period we are trying to predict. It is like seeing the exam answers. This is called "leakage".
- I removed them. With them, the score would look better than it really is: 0.80 would become 0.87. An automatic test makes sure this cannot happen again.

**Point at:** the orange dots on the chart. These are the broken spend values.

**Next slide:** Once the data was clean, one chart changed my whole approach.

---

## Slide 4: Two different churn problems hide inside one number

*About 53 seconds*

**Main point:** There are two different kinds of churn hidden in one number.

**Say:**

- This chart shows churn by "months since the member's last purchase".
- Bought last month: about 3% leave. Missed 1 month: 38%. Missed 2 months: 81%. Missed 3 or more: almost 100%.
- So the official churn rate of 34% mixes two groups:
- LAPSED: 653 members who were already gone before April. A simple rule finds them.
- ACTIVE: 1,715 members still shopping. Only 9.6% of them leave. This is where prediction really matters.
- Any model looks brilliant on the lapsed group. So from now on I show results for ACTIVE members separately. That is the honest number.

**Point at:** the blue bars (active) and then the orange bars (lapsed).

**Next slide:** So what happens before an active member leaves?

---

## Slide 5: Churn has a fingerprint, and a two-month window to act

*About 55 seconds*

**Main point:** Leaving has warning signs, about 2 months in advance.

**Say:**

- I lined up every active member on their last purchase, and looked back 6 months.
- Blue = members who stayed. Their behaviour is flat.
- Orange = members who left. On average they buy less, use the app less and open fewer emails, and their support tickets jump.
- This gives the business a 2-month window to act.
- Three simple alarms that need no model:
- spend drops by 40% or more: 37% of these members leave.
- app used less than once every 2 months: 46% leave.
- a formal complaint: 17% leave, compared with 9%.
- All of these are statistically significant.

**Point at:** the "Support tickets" panel. The orange line shoots up at the end.

**Next slide:** Which signals matter most, and does membership tier protect us?

---

## Slide 6: Support experience and engagement matter. Tier does not.

*About 43 seconds*

**Main point:** Service problems and low engagement matter. Membership tier does not.

**Say:**

- The brief asked about three things: tier, marketing engagement and support experience.
- Support: YES. 3 or more tickets means 37% churn, compared with 4%.
- Engagement: YES. App use and email opens are among the strongest signals in both models.
- Tier: NO. Platinum members leave as often as Silver members (34% vs 35%). The expensive perks are not creating loyalty.
- The bar at the bottom shows where the model's signal comes from: mostly behaviour and engagement.

**Point at:** the "Tier" bullet on the right.

**Next slide:** Before trusting any model, I tested it the honest way.

---

## Slide 7: Validated the way it will be used: train on the past, test on an unseen quarter

*About 58 seconds*

**Main point:** I tested the model the way it will really be used: learn from the past, predict the future.

**Say:**

- A random split mixes past and future data. The results look better than they really are.
- Instead I trained on July and October 2023, tuned on January 2024, and tested ONCE on the real April-June 2024 answers.
- The model only sees the 6 months before each date. It never sees the future.
- The scores are calibrated: "30% risk" really means about 30 in 100 such members leave. This matters for the money calculations later.
- Engineering: every model run is tracked in MLflow, and the feature code also runs in Spark for Databricks. I tested that it gives identical results.

**Point at:** the three boxes at the bottom. The honest test gives 0.81. The "cheating" columns would have given 0.87.

**Next slide:** So how good is the model?

---

## Slide 8: On Active members, machine learning roughly doubles what a recency rule can do

*About 60 seconds*

**Main point:** For active members, machine learning is about 2x better than a simple rule.

**Say:**

- A simple rule ("has not bought recently") scores 0.46.
- My machine-learning models score about 0.8. (1.0 is perfect. Random guessing would score about 0.10.)
- Logistic regression won on the tuning data, so I chose it. It is simple and easy to explain. Gradient boosting is about the same.
- At my chosen threshold the model flags 179 members, and it catches 3 out of 4 real leavers.
- The calibration chart (bottom right) shows the risk percentages can be trusted.
- I also tried a deep-learning model (a GRU). It scores slightly higher (0.82). My plan: run it in the background for one quarter, and switch only if it stays better.

**Point at:** the table. The bold row is the model I chose.

**Next slide:** A score alone is not enough. People need to know WHY.

---

## Slide 9: Every member gets a score and three reasons the CRM team can act on

*About 57 seconds*

**Main point:** Every member gets a risk score, three reasons, and a ready-to-send message.

**Say:**

- Left: what pushes risk up for the whole group: low purchases, low app and email use, and recent support tickets.
- Right: what the store team sees for EACH member: the risk, plus 3 reasons in plain English.
- One honest caution: these reasons explain the model. They do not prove cause. That is why I recommend a test before spending at scale.
- The last step is automated too. An AI writes the message or call script from the same 3 reasons.
- Safety checks block made-up promo codes, wrong apologies and any mention of tracking. If the AI fails a check, a safe template is used.

**Point at:** one member card on the right.

**Next slide:** Do we really need behaviour data, or is basic customer information enough?

---

## Slide 10: Behaviour predicts churn. Demographics alone are no better than chance.

*About 50 seconds*

**Main point:** Behaviour predicts churn. Age, city and tier alone do not.

**Say:**

- I added the data groups one at a time (the bars on the right).
- Demographics only: 0.11. That is no better than guessing.
- Add purchase behaviour: 0.66. Add engagement: 0.77. Add support: 0.80.
- Even without "months since last purchase", the model still scores about 0.8. So it finds real early warning signs.
- Left chart, by city and tier: the differences are small. Austin and Columbus are the highest; members who joined in 2024 are the lowest.
- So use segments to decide WHO acts (which store team), not WHO to target.

**Next slide:** Now the most important part for the business: the money.

---

## Slide 11: Targeting turns retention from a cost into a return

*About 63 seconds*

**Main point:** Targeting turns retention from a loss into a profit.

**Say:**

- My assumptions, stated clearly: a $10 coupon lowers a member's churn risk by 15%. A $15 service call lowers it by 25%. A member's value is 6 months of their spending, at a 25% margin.
- Today's coupon for everyone costs about $24k and loses $15.4k. Most people were staying anyway, or had already left.
- My plan gives each member at most one action, and only when it pays back. About 43 members are kept, for $3.9k. That is a $3.6k profit.
- Calling everyone with support tickets loses money. Calling only the members the model flags makes money.
- Targeting still breaks even if the offers work only about 60% as well as I assumed. The coupon for everyone never breaks even.

**Point at:** the red bar (today's approach), then the green bar (my plan).

**Next slide:** So here is what I recommend.

---

## Slide 12: Recommendations: a 90-day plan

*About 56 seconds*

**Main point:** Six actions for the next 90 days.

**Say:**

1. Report the two kinds of churn separately, and add an alert after 60 days with no purchase.
2. Score every member each month, and give each one the single best action.
3. Fix support problems fast, inside the 2-month warning window.
4. Try free nudges (app and email) before paid offers.
5. Rethink tier perks. Platinum is not buying loyalty.
6. Prove it before scaling: keep 20% of flagged members as a control group. About 729 flagged members are enough to measure a 15% effect.
- Everything rebuilds with one command, and there is a live dashboard for the CRM team.

> **Optional:** about 60 seconds, only if time allows): open the live dashboard, click a high-risk member, click "Draft with AI", then move one slider in the simulator.
- Thank you. I am happy to take questions.

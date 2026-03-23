# What to do

We have an incredible plan for optimizing our hot path in the SDDP algorithm in plans/lp-setup-optimization . We want to be absolutely sure that it makes sense and won't break any of our requirements for the SDDP implementation. Moreover, we want a detailed explanation about why each implementation will probably result in the expected performance gains, highlighting the identified bottlenecks that motivated the implementations.

# How to proceed

You do this in a loop and update an assessment document plan-review-history.md here in the repo root.

If you notice any improvement suggestions to the plan, update it, and document what you did in the plan-review-history.md.

## For each step of the plan

    - You begin reading each step of the plan (epic/ticket).
    - You consult the original spec file if needed
    - You read relevant parts of the code that might be associated with it
    - You check in the cobre and cobre-docs documentation for related parts to it
    - You consider our major requirements for the cobre implementation (bit-by-bit reproducibility, architectural quality, subcrate independence)
    - You evaluate if the proposed actions in the plan ACTUALLY AIM to solve the problems that we observed in our tests
    - If so, you write a detailed explanation of why in plan-review-history.md
    - If not, you think DEEPLY on a better solution and update the plan, writing in plan-review-history.md what you changed and why this works now

# At the end

Write a summary at the end of plan-review-history.md with your conclusions
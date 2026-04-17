# Correlation Logic for Trace Analysis

First we get the commit history as a list of Candidate objects from the GitWorkspace.
In the ExperimentDatabase, we can use `get_experiments` to get a list of Experiment objects.
Each Experiment object has a run, which is an ExperimentRun. ExperimentRun has a candidate, which is a Candidate object.
Convert this into dict[Candidate, list[Experiment]], since a candidate can be evaluated multiple times.

Now we can iterate over the commit history and check if the candidate has been evaluated. If it has, we create a cut.
At the end of this, we should have a list[GitCommitHistory], where the final element is a candidate that has been evaluated (except for the last phase).

This is a disjoint representation and what we mean by OptimizationPhase. Each OptimizationPhase will have a final candidate
but also an initial candidate.

So now imagine we have a sequence like:
[(phase_0_initial_candidate, phase_0_final_candidate), (phase_1_initial_candidate, phase_1_final_candidate), ...]

How do we figure out the relevant trace span segments from the agent trace for each optimization phase?

When do commit phase changes happen in our agent trace?
These are on file writes/edits, resource edits, and git restores.
The agent trace is a sequence of items. For each of the above types, we should identify the indices of the items
in the trace that match a type. Then we need to figure out which commit that item is associated with.
For file/resource edits, we need to extract the commit from the **tool response**.
For git restores, we need to extract the commit of the tool response of the git restore tool call.
We need to ensure that the tool call was successful, i.e. the response was not an error.

Now we have a list of list of spans: list[list[span]], each ending on a commit change.
The sequence of commits must be ordered in both lists. so now need to correlate them.

we can greedily match here: create two pointers. iterate over the list of list of spans, if the commit
does not match the pointer to the final commit of the optimization phase, add it to the optimization phase.
if it does match, add it and then increment both pointers. In this way, each optimization phase will have a list of list of spans associated with it.

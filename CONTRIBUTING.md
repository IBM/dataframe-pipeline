# Contributing

This is an open source project, and we appreciate your help!

We use the [GitHub issue tracker](https://github.com/IBM/dataframe-pipeline/issues) to discuss new features and non-trivial bugs.

This project requires signing [DCO (Developer Certificate of Origin, Version 1.1)](https://developercertificate.org/) to make any contribution. When you want to make a contribution, please make a commit **with ```-s``` option** like ```git commit . -s -m "commit message"```. The ```-s``` option appends a ```Signed-off-by``` line to your commit message like ```Signed-off-by: Random J Developer <random@developer.example.org>```. If all of the commits in your pull request are signed (i.e., your pull request is valid), you are asked to sign DCO. This automation is done by using a [cla-assistant](https://cla-assistant.io/) and [DCO bot](https://github.com/probot/dco).

If one or more commits in a pull request are not signed, you are asked to sign all of your commits. Again, **every commit** in a pull request needs to be signed. If you forget to sign one or more commits, please use the following commands to squash the old PR (original branch) into a single commit. 

```
git checkout master
git checkout -b new_branch                   # create a new branch as temporary
git merge --squash your_original_branch      # copy from your original branch
git branch -d your_original_branch           # remove your original branch
git checkout -b your_original_branch         # create a new branch with the same name (override)
git commit -m 'type your own commit msg' -s  # signoff that single commit
git push origin your_original_branch -f      # forcibly override your original branch`
```

Generally, we expect two maintainers to review your pull request before it is approved for merging.

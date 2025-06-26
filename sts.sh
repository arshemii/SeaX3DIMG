#!/bin/bash

# A script to auto-commit and push all changes to the current Git branch

# Step 1: Print status and branch
echo "🔍 Checking current branch and status..."
branch=$(git branch --show-current)
echo "📌 Current branch: $branch"
git status

# Step 2: Stage all changes (including new and deleted files)
echo "🌀 Staging all changes..."
git add .

# Step 3: Commit with a timestamped message
commit_msg="Auto-commit on $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$commit_msg"

# Step 4: Print confirmation
echo "✅ Commit completed with message: '$commit_msg'"

# Step 5: Push to remote branch
echo "🚀 Pushing to GitHub..."
git push origin ImgOnly

# Step 6: Show final log
echo "📜 Latest commit log:"
git log -1 --oneline

# Final message
echo "✅ Done! All changes are pushed to GitHub on branch '$branch'."


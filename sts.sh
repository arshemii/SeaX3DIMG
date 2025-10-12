#!/bin/bash

# A script to auto-commit and push all changes to GitHub with pull safety

echo "🔍 Checking current branch..."
branch=$(git branch --show-current)
echo "📌 Current branch: $branch"

# Show current status
git status

# Step 1: Stage all changes
echo "🌀 Staging all changes..."
git add .

# Step 2: Commit with timestamp
commit_msg="Auto-commit on $(date '+%Y-%m-%d %H:%M:%S')"
git commit -m "$commit_msg" 2>/dev/null

# Check if commit was successful (to avoid empty commits)
if [ $? -eq 0 ]; then
  echo "✅ Commit completed with message: '$commit_msg'"
else
  echo "⚠️ No changes to commit."
fi

# Step 3: Pull before push (merge if needed)
echo "📥 Pulling latest from origin/ImgOnly_LT..."
git pull origin "$branch" --rebase

# Step 4: Push changes
echo "🚀 Pushing to GitHub..."
git push origin "ImgOnly_LT"

# Step 5: Show latest commit
echo "📜 Latest commit log:"
git log -1 --oneline

echo "✅ All done on branch 'ImgOnly_LT'!"


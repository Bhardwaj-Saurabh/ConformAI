#!/bin/bash

# Script to remove data directory from git tracking
# This keeps your local data files but removes them from version control

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         Removing Data Directory from Git Tracking                ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

echo "✓ Git repository found"
echo ""

# Check current git status
echo "Current git status:"
echo "─────────────────────────────────────────────────────────────────"
git status --short
echo "─────────────────────────────────────────────────────────────────"
echo ""

# Remove data directory from git tracking (but keep local files)
echo "Removing data/ from git tracking (keeping local files)..."
if git ls-files data/ | grep -q .; then
    git rm -r --cached data/
    echo "✓ Removed data/ from git index"
else
    echo "ℹ️  data/ directory is not tracked in git"
fi

# Remove airflow logs if tracked
echo ""
echo "Removing airflow/logs/ from git tracking (keeping local files)..."
if git ls-files airflow/logs/ | grep -q . 2>/dev/null; then
    git rm -r --cached airflow/logs/
    echo "✓ Removed airflow/logs/ from git index"
else
    echo "ℹ️  airflow/logs/ directory is not tracked in git"
fi

# Remove any .log files if tracked
echo ""
echo "Removing *.log files from git tracking..."
if git ls-files '*.log' | grep -q . 2>/dev/null; then
    git ls-files '*.log' | xargs git rm --cached
    echo "✓ Removed .log files from git index"
else
    echo "ℹ️  No .log files tracked in git"
fi

echo ""
echo "─────────────────────────────────────────────────────────────────"
echo "New git status:"
echo "─────────────────────────────────────────────────────────────────"
git status --short
echo "─────────────────────────────────────────────────────────────────"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                     Next Steps                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "The data/ directory and log files have been removed from git tracking."
echo "Your local files are still safe and intact."
echo ""
echo "To complete the process:"
echo ""
echo "1. Review the changes:"
echo "   git status"
echo ""
echo "2. Commit the changes:"
echo "   git add .gitignore"
echo "   git commit -m \"Remove data directory and logs from version control"
echo ""
echo "   - Add data/, airflow/logs/, and *.log to .gitignore"
echo "   - Remove tracked data files from git index"
echo "   - Keep local data files intact\""
echo ""
echo "3. Push to GitHub:"
echo "   git push origin main"
echo ""
echo "Note: If you've already pushed data/ to GitHub, this will remove it"
echo "from future commits but not from the git history. To completely remove"
echo "sensitive data from git history, you'll need to use git filter-branch"
echo "or BFG Repo-Cleaner (ask if you need help with this)."
echo ""

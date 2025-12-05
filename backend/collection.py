import json
import os
import time
from github import Github, GithubException
from tqdm import tqdm
from config import GITHUB_TOKEN, TARGET_REPOS, DATA_DIR

class GitHubDataCollector:
    def __init__(self):
        if not GITHUB_TOKEN:
            raise ValueError("❌ Missing GITHUB_TOKEN in .env file")
        self.gh = Github(GITHUB_TOKEN)
        self.data_dir = DATA_DIR

    def extract_repo_data(self, repo_name):
        """Extract PRs and Issues from a single repo"""
        print(f"\n🔄 Connecting to {repo_name}...")
        
        try:
            repo = self.gh.get_repo(repo_name)
            
            # 1. FETCH PULL REQUESTS
            print(f"   📥 Fetching Pull Requests...")
            prs_data = []
            # Fetch last 100 PRs (closed ones are best for training)
            pulls = repo.get_pulls(state='closed', sort='updated', direction='desc')[:100]
            
            for pr in tqdm(pulls, desc="   Downloading PRs", total=100):
                try:
                    pr_item = {
                        "id": pr.number,
                        "title": pr.title,
                        "body": pr.body,
                        "state": pr.state,
                        "merged": pr.merged,
                        "created_at": pr.created_at.isoformat(),
                        "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
                        "user": pr.user.login if pr.user else "ghost",
                        "files": [f.filename for f in pr.get_files()],
                        "repo": repo_name
                    }
                    prs_data.append(pr_item)
                except Exception as e:
                    continue # Skip if error on specific PR

            # 2. FETCH ISSUES (User Stories)
            print(f"   📥 Fetching Issues...")
            issues_data = []
            # Fetch last 100 Issues
            issues = repo.get_issues(state='closed', sort='updated', direction='desc')[:100]
            
            for issue in tqdm(issues, desc="   Downloading Issues", total=100):
                if issue.pull_request: continue # Skip PRs (they appear in issues API)
                
                try:
                    issue_item = {
                        "id": issue.number,
                        "title": issue.title,
                        "body": issue.body,
                        "labels": [l.name for l in issue.labels],
                        "state": issue.state,
                        "repo": repo_name
                    }
                    issues_data.append(issue_item)
                except Exception:
                    continue

            return prs_data, issues_data

        except GithubException as e:
            print(f"❌ Error Accessing {repo_name}: {e}")
            return [], []

    def run(self):
        all_prs = []
        all_issues = []

        for repo_name in TARGET_REPOS:
            prs, issues = self.extract_repo_data(repo_name)
            all_prs.extend(prs)
            all_issues.extend(issues)
            print(f"   ✅ {repo_name}: Found {len(prs)} PRs, {len(issues)} Issues")

        # SAVE TO FILE
        self.save_json(all_prs, "prs.json")
        self.save_json(all_issues, "issues.json")

    def save_json(self, data, filename):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\n💾 Saved {len(data)} items to {filepath}")

if __name__ == "__main__":
    collector = GitHubDataCollector()
    collector.run()

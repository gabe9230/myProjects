import sys
import subprocess
import importlib

# === Auto-install missing modules ===
def install_and_import(package):
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)

yaml = install_and_import("yaml")
requests = install_and_import("requests")
bs4 = install_and_import("bs4")
from bs4 import BeautifulSoup
PyQt5 = install_and_import("PyQt5")
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QListWidget,
    QHBoxLayout, QMessageBox
)

# === Load Config ===
def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# === Job Scrapers ===

def scrape_jobicy_jobs():
    jobs = []
    try:
        url = "https://jobicy.com/api/v2/remote-jobs"
        res = requests.get(url, timeout=1)
        if res.status_code == 200:
            data = res.json()
            for post in data.get("jobs", []):
                job = {
                    "title": post.get("title", "").strip(),
                    "company": post.get("company", "Jobicy"),
                    "location": post.get("location", "Remote"),
                    "link": post.get("url", ""),
                    "score": 0,
                    "source": "Jobicy"
                }
                jobs.append(job)
    except Exception as e:
        print(f"[!] Scrape error (Jobicy): {e}")
    print(f"[+] Jobicy: {len(jobs)} jobs scraped")
    return jobs

import json

with open("top500_companies.json", "r") as f:
    top500 = json.load(f)

def scrape_lever_jobs():
    jobs = []
    try:
        companies = top500.get("lever", [])  # example public boards
        for company in companies:
            url = f"https://api.lever.co/v0/postings/{company}"
            res = requests.get(url, timeout=1)
            if res.status_code != 200:
                continue
            data = res.json()
            for post in data:
                    categories = post.get("categories", {}).values()
                    location_str = ", ".join(
                        loc["location"] if isinstance(loc, dict) and "location" in loc else str(loc)
                        for loc in categories
                    )
                    job = {
                        "title": post.get("text", "").strip(),
                        "company": company.title(),
                        "location": location_str,
                        "link": post.get("hostedUrl", ""),
                        "score": 0,
                        "source": "Lever"
                    }
                    jobs.append(job)
    except Exception as e:
        print(f"[!] Scrape error (Lever): {e}")
    return jobs

def scrape_workable_jobs():
    jobs = []
    try:
        companies = top500.get("workable", [])  # example company tokens
        for token in companies:
            url = f"https://apply.workable.com/api/v1/jobs?token={token}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                data = res.json()
                for post in data.get("jobs", []):
                    job = {
                        "title": post.get("title", "").strip(),
                        "company": token,
                        "location": post.get("location", "Remote"),
                        "link": post.get("url", ""),
                        "score": 0,
                        "source": "Workable"
                    }
                    jobs.append(job)
    except Exception as e:
        print(f"[!] Scrape error (Workable): {e}")
    return jobs

def scrape_greenhouse_jobs():
    jobs = []
    try:
        urls = top500.get("greenhouse", [])
        for url in urls:
            res = requests.get(url, timeout=5)
            if res.status_code != 200:
                continue
            data = res.json()
            for post in data.get("jobs", []):
                job = {
                    "title": post.get("title", "").strip(),
                    "company": "Greenhouse",
                    "location": post.get("location", {}).get("name", "Remote"),
                    "link": post.get("absolute_url", ""),
                    "score": 0,
                    "source": "Greenhouse"
                }
                jobs.append(job)
    except Exception as e:
        print(f"[!] Scrape error (Greenhouse): {e}")
    return jobs

def scrape_usajobs():
    jobs = []
    try:
        url = "https://data.usajobs.gov/api/search?Keyword=machinist"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            for post in data.get("SearchResult", {}).get("SearchResultItems", []):
                fields = post.get("MatchedObjectDescriptor", {})
                job = {
                    "title": fields.get("PositionTitle", "").strip(),
                    "company": fields.get("OrganizationName", "USAJOBS"),
                    "location": fields.get("PositionLocation", [{}])[0].get("LocationName", "Unknown"),
                    "link": fields.get("PositionURI", ""),
                    "score": 0,
                    "source": "USAJOBS"
                }
                jobs.append(job)
    except Exception as e:
        print(f"[!] Scrape error (USAJOBS): {e}")
    return jobs

indeed_blocked = False


def scrape_remotive_jobs(query="cnc machinist", limit=100):
    jobs = []
    try:
        url = "https://remotive.com/api/remote-jobs"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        print(f"[DEBUG] GET {url} => {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            for post in data.get("jobs", []):
                try:
                    title = post.get("title", "").strip()
                    company = post.get("company_name", "").strip()
                    location = post.get("candidate_required_location", "Remote")
                    link = post.get("url", "")
                    if title and company and link:
                        job = {
                            "title": title,
                            "company": company,
                            "location": location,
                            "link": link,
                            "score": 0,
                            "source": "Remotive"
                        }
                        jobs.append(job)
                        
                except Exception as e:
                    print(f"[!] Remotive job parse error: {e}")
    except Exception as e:
        print(f"[!] Scrape error (Remotive): {e}")
    return jobs

def scrape_remoteok_jobs(query="cnc machinist", limit=100):
    jobs = []
    try:
        url = "https://remoteok.com/api"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        print(f"[DEBUG] GET {url} => {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            for post in data[1:]:  # Skip the metadata entry
                try:
                    title = post.get("position", "").strip()
                    company = post.get("company", "").strip()
                    location = post.get("location", "Remote")
                    link = "https://remoteok.com" + post.get("url", "")
                    if title and company and link:
                        job = {
                            "title": title,
                            "company": company,
                            "location": location,
                            "link": link,
                            "score": 0,
                            "source": "RemoteOK"
                        }
                        jobs.append(job)
                        if len(jobs) >= limit:
                            break
                except Exception as e:
                    print(f"[!] RemoteOK job parse error: {e}")
    except Exception as e:
        print(f"[!] Scrape error (RemoteOK): {e}")
    return jobs

# === Score Jobs ===
def score_jobs(jobs, config):
    scored = []
    priority_keywords = config.get("priority_keywords", {})
    exclude_keywords = config.get("exclude_keywords", [])

    for job in jobs:
        score = 0
        title_lower = (job["title"] + " " + job["company"]).lower()
        for kw, weight in priority_keywords.items():
            if kw.lower() in title_lower:
                score += weight
        for ex in exclude_keywords:
            if ex.lower() in title_lower:
                score -= 2  # Penalty weight for excluded terms
        job["score"] = round(score, 2)
        scored.append(job)

    return sorted(scored, key=lambda x: -x["score"])

# === Main GUI ===
class JobApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Job Application Assistant")
        self.layout = QVBoxLayout()

        import time
        start_time = time.time()
        loading_label = QLabel("Loading jobs... please wait")
        self.layout.addWidget(loading_label)
        self.repaint()

        jobs_raw = scrape_remoteok_jobs() + scrape_remotive_jobs() + scrape_greenhouse_jobs() + scrape_usajobs() + scrape_lever_jobs() + scrape_workable_jobs() + scrape_jobicy_jobs()
        print(f"[*] Scraped {len(jobs_raw)} jobs in {round(time.time() - start_time, 2)}s")
        self.jobs = score_jobs(jobs_raw, config)
        self.layout.removeWidget(loading_label)
        loading_label.deleteLater()

        self.job_list = QListWidget()
        for job in self.jobs:
            if job["score"] >= 1:
                self.job_list.addItem(f"{job['title']} – {job['company']} ({job['score']}★) [{job.get('source', 'Unknown')}]")
        self.layout.addWidget(self.job_list)

        # Buttons
        self.button_layout = QHBoxLayout()
        self.open_button = QPushButton("Open Link")
        self.apply_button = QPushButton("Auto-Fill + Verify")
        self.skip_button = QPushButton("Skip")
        self.blacklist_button = QPushButton("Blacklist Company")

        for btn in [self.open_button, self.apply_button, self.skip_button, self.blacklist_button]:
            self.button_layout.addWidget(btn)
        self.layout.addLayout(self.button_layout)

        self.setLayout(self.layout)

        # Connect buttons
        self.open_button.clicked.connect(self.open_link)
        self.apply_button.clicked.connect(self.autofill_form)
        self.skip_button.clicked.connect(self.skip_job)
        self.blacklist_button.clicked.connect(self.blacklist_company)

    def current_job(self):
        idx = self.job_list.currentRow()
        return self.jobs[idx] if idx >= 0 else None

    def open_link(self):
        job = self.current_job()
        if job:
            import webbrowser
            webbrowser.open(job['link'])

    def autofill_form(self):
        job = self.current_job()
        if job:
            confirm = QMessageBox.question(self, "Confirm", f"Auto-fill and submit application to {job['company']}?",
                                           QMessageBox.Yes | QMessageBox.No)
            if confirm == QMessageBox.Yes:
                # Placeholder for actual automation
                QMessageBox.information(self, "Submitted", f"Application sent to {job['company']}")

    def skip_job(self):
        idx = self.job_list.currentRow()
        if idx >= 0:
            self.job_list.takeItem(idx)

    def blacklist_company(self):
        job = self.current_job()
        if job:
            # Save to blacklist (this is a stub)
            QMessageBox.information(self, "Blacklisted", f"{job['company']} has been blacklisted.")
            self.skip_job()

# === Main Entry ===
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = JobApp()
    window.show()
    sys.exit(app.exec_())

"""
Jira Selenium Connector for fetching task details from JavaScript-rendered pages.
Use this when Jira pages require JavaScript execution to display content.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import logging
from typing import Tuple


class JiraSeleniumConnector:
    """Fetches Jira ticket details using Selenium for JavaScript-rendered pages."""

    def __init__(self, jira_url: str):
        """
        Initialize the Jira Selenium connector.

        Args:
            jira_url: Base URL of the Jira instance (e.g., https://issues.apache.org/jira)
        """
        self.jira_url = jira_url.rstrip('/')
        self.logger = logging.getLogger(__name__)

    def fetch_task_details(self, task_key: str) -> Tuple[str, str, str]:
        """
        Fetch task details from Jira using Selenium.

        Args:
            task_key: Jira task key (e.g., JIRA-123)

        Returns:
            Tuple of (title, description, comments)
        """
        comment_tab = '?page=com.atlassian.jira.plugin.system.issuetabpanels:comment-tabpanel'
        task_url = f"{self.jira_url}/browse/{task_key}{comment_tab}"

        browser = None
        try:
            # Configure Chrome options for headless mode
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--blink-settings=imagesEnabled=false')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')

            browser = webdriver.Chrome(options=options)
            browser.get(task_url)

            # Wait for page to load
            browser.implicitly_wait(10)

            # Parse the HTML
            soup = BeautifulSoup(browser.page_source, 'html.parser')

            # Extract title
            title_element = soup.find('h1', {'id': 'summary-val'})
            title = title_element.text.strip() if title_element else ""

            # Extract description
            description_element = soup.find('div', {'id': 'description-val'})
            description = description_element.text.strip() if description_element else ""

            # Extract comments
            comment_elements = soup.find_all(class_='twixi-wrap concise actionContainer')
            comments = [comment.text.strip() for comment in comment_elements]

            return title, description, str(comments)

        except Exception as e:
            self.logger.error(f"Error fetching task {task_key}: {e}")
            return "", "", ""
        finally:
            if browser:
                browser.quit()

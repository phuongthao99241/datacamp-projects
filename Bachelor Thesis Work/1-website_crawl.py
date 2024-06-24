import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from retry import retry

# Status: 01/10/2023
# The scraping Python is created based on the learning from various online resources, tutorials, and documentation related to Python web scraping.

# Define user-agent header
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# Define the base URL
base_url = "https://careerbuilder.vn/viec-lam/tat-ca-viec-lam-trang-{}-vi.html"

# Define page range to crawl data
page_number = 1
max_page = 15
jobs = []


# Define a function to make HTTP requests with retry
@retry(tries=3, delay=2)
def get_response(url):
    return requests.get(url, headers=headers, timeout=30)


# Loop through pages
while page_number < max_page:
    URL = base_url.format(page_number)

    try:
        # Get the page content
        response = get_response(URL)
    except requests.exceptions.ChunkedEncodingError:
        print(f"ChunkedEncodingError for URL: {URL}")
        page_number += 1
        continue

    # Check status code to ensure that the HTTP request has succeeded
    if response.status_code != 200:
        print(f"Error fetching page {URL}, status code: {response.status_code}")
        break

    # Create a BeautifulSoup object to parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Collect all jobs shown on each page
    job_items = soup.find_all('div', class_='job-item')

    if not job_items:
        break

    for job_item in job_items:
        # Look up job titles and job links
        job_title = job_item.find('a', class_='job_link').text.strip()
        detailed_job_url = job_item.find('a', class_='job_link')['href']

        job_info = {
            'Title': job_title,
            'Detailed Job URL': detailed_job_url
        }

        # Access job through job link
        try:
            # Get the detailed job page content after accessing link
            detailed_response = get_response(detailed_job_url)
        except requests.exceptions.ChunkedEncodingError:
            print(f"ChunkedEncodingError for URL: {detailed_job_url}")
            continue

        # Create a BeautifulSoup object to parse the HTML content of the page
        detailed_soup = BeautifulSoup(detailed_response.content, 'html.parser')

        # Extract job details
        detail_div = detailed_soup.find('div', class_='detail-row reset-bullet')
        if detail_div:
            job_details = [li.text.strip() for li in detail_div.find_all('li')]
            job_info['Job Details'] = job_details
        else:
            job_info['Job Details'] = []

        # Extract job requirements ("Yêu Cầu Công Việc" means job requirements)
        requirements_title = detailed_soup.find('h2', class_='detail-title', string='Yêu Cầu Công Việc')
        if requirements_title and requirements_title.parent:
            requirements_div = requirements_title.parent
            job_requirements = [li.text.strip() for li in requirements_div.find_all('li')]
            job_info['Job Requirements'] = job_requirements
        else:
            job_info['Job Requirements'] = []

        jobs.append(job_info)

    # Pause for 2 seconds before fetching the next page to avoid making requests too quickly
    time.sleep(2)

    # Repeat process with next page
    page_number += 1

# Create a DataFrame and save the data to a CSV file
df = pd.DataFrame(jobs)
df.to_csv('jobs-website.csv', index=False)

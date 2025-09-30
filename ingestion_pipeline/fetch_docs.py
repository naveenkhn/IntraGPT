from atlassian import Confluence
import os
import json
import re

# Replace with your service account details
CONFLUENCE_URL = "https://org_name.atlassian.net/wiki"
USERNAME = "robotic_account_mail_id"
API_TOKEN = "token_value"

SPACE_KEY = "SPACE_NAME"

confluence = Confluence(
    url=CONFLUENCE_URL,
    username=USERNAME,
    password=API_TOKEN,
    timeout=300
)

OUTPUT_DIR = "confluence_export"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

import time
import logging
from requests.exceptions import Timeout, HTTPError

def save_page_content(page, folder_path):
    page_id = page['id']
    title = page['title']
    safe_title = safe_filename(title)

    json_path = os.path.join(folder_path, f"{safe_title}_{page_id}.json")
    html_path = os.path.join(folder_path, f"{safe_title}_{page_id}.html")

    # Check if JSON exists and has lastModified
    existing_last_modified = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f_json:
                existing_meta = json.load(f_json)
                existing_last_modified = existing_meta.get("lastModified")
        except Exception as e:
            logging.warning(f"Could not read {json_path}: {e}")

    # Fetch full page with version + html body, with retry and exception handling
    max_retries = 3
    delay = 2
    page_content = None
    for attempt in range(max_retries):
        try:
            page_content = confluence.get_page_by_id(page_id, expand='body.export_view,version,ancestors')
            break
        except (Timeout, HTTPError) as e:
            logging.error(f"Error fetching page ID {page_id} on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                # Save minimal metadata and skip content saving
                metadata = {
                    "id": page_id,
                    "title": title,
                    "space": SPACE_KEY,
                    "url": f"{CONFLUENCE_URL}/spaces/{SPACE_KEY}/pages/{page_id}",
                    "lastModified": None,
                    "parentId": None,
                    "error": f"Failed to fetch page after {max_retries} attempts: {str(e)}"
                }
                with open(json_path, 'w', encoding='utf-8') as f_json:
                    json.dump(metadata, f_json, indent=2)
                print(f"Skipped page ID {page_id} titled '{title}' due to repeated errors.")
                return
        except Exception as e:
            logging.error(f"Unexpected error fetching page ID {page_id}: {e}")
            # Save minimal metadata and skip content saving
            metadata = {
                "id": page_id,
                "title": title,
                "space": SPACE_KEY,
                "url": f"{CONFLUENCE_URL}/spaces/{SPACE_KEY}/pages/{page_id}",
                "lastModified": None,
                "parentId": None,
                "error": f"Unexpected error: {str(e)}"
            }
            with open(json_path, 'w', encoding='utf-8') as f_json:
                json.dump(metadata, f_json, indent=2)
            print(f"Skipped page ID {page_id} titled '{title}' due to unexpected error.")
            return

    # At this point, page_content is present
    page_last_modified = page_content.get('version', {}).get('when')
    # If JSON exists and lastModified matches, skip re-downloading
    if existing_last_modified and page_last_modified and existing_last_modified == page_last_modified:
        print(f"Page ID {page_id} titled '{title}' already up-to-date, skipping.")
        return

    html_content = page_content.get('body', {}).get('export_view', {}).get('value', '')

    # Save HTML
    with open(html_path, 'w', encoding='utf-8') as f_html:
        f_html.write(html_content)

    # Extract metadata
    metadata = {
        "id": page_id,
        "title": title,
        "space": SPACE_KEY,
        "url": f"{CONFLUENCE_URL}/spaces/{SPACE_KEY}/pages/{page_id}",
        "lastModified": page_last_modified,
        "parentId": page_content['ancestors'][-1]['id'] if page_content.get('ancestors') else None
    }

    # Save metadata JSON
    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(metadata, f_json, indent=2)

    print(f"Exported page ID {page_id} titled '{title}'")

def fetch_and_save_page(page, parent_folder):
    page_id = page['id']
    title = page['title']
    safe_title = safe_filename(title)
    page_folder = os.path.join(parent_folder, f"{safe_title}_{page_id}")
    os.makedirs(page_folder, exist_ok=True)

    # Save page + metadata
    save_page_content(page, page_folder)

    # Fetch children and recurse
    children = confluence.get_page_child_by_type(page_id, type='page')
    for child in children:
        fetch_and_save_page(child, page_folder)

def main():
    start = 0
    limit = 50
    while True:
        pages = confluence.get_all_pages_from_space(
            space=SPACE_KEY,
            start=start,
            limit=limit,
            status=None,
            expand='ancestors,version',
            content_type='page'
        )
        if not pages:
            break
        for page in pages:
            if not page.get('ancestors'):  # only top-level
                fetch_and_save_page(page, OUTPUT_DIR)
        start += limit

if __name__ == "__main__":
    main()
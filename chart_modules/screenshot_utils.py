from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from PIL import Image
import time
import tempfile
import random
import os
import sys

def get_driver(max_retries=1, delay=0):
    """
    Start a stable headless Chrome instance, with Linux headless environment support.
    """
    for attempt in range(1, max_retries + 1):
        try:
            options = webdriver.ChromeOptions()
            # Headless + software rendering for stable startup on Linux
            options.add_argument("--headless=new")   # new headless mode
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-software-rasterizer")
            options.add_argument("--disable-gpu")
            # options.add_argument("--remote-debugging-port=9222")
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-background-networking")
            options.add_argument("--disable-default-apps")
            options.add_argument("--disable-sync")
            options.add_argument("--disable-translate")
            options.add_argument("--metrics-recording-only")
            options.add_argument("--mute-audio")

            unique_tmpdir = tempfile.mkdtemp(prefix="chrome_cache_")
            cache_dir = os.path.join(unique_tmpdir, "cache")
            crash_dir = os.path.join(unique_tmpdir, "crash")
            media_cache_dir = os.path.join(unique_tmpdir, "media_cache")
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(crash_dir, exist_ok=True)
            os.makedirs(media_cache_dir, exist_ok=True)

            options.add_argument(f"--disk-cache-dir={cache_dir}")
            options.add_argument(f"--media-cache-dir={media_cache_dir}")
            options.add_argument(f"--crash-dumps-dir={crash_dir}")

            service = ChromeService(
                log_output=open(f"chromedriver_attempt{attempt}.log", "w")
            )
            # service = ChromeService(log_output=sys.stdout)  # print chromedriver log
            options.add_argument("--enable-logging")
            options.add_argument("--v=1")
            options.add_argument("--log-file=chrome.log")   # print Chrome log

            unique_tmpdir = tempfile.mkdtemp(prefix="chrome_")
            options.add_argument(f"--user-data-dir={unique_tmpdir}")

            driver = webdriver.Chrome(options=options)
            print(f">>> ChromeDriver started successfully on attempt {attempt}")
            return driver
        except WebDriverException as e:
            if attempt < max_retries:
                print(f"[Retry {attempt}/{max_retries}] Chrome failed to start: {e}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise  # Exceeded max retries, raise exception


def take_screenshot(driver: webdriver.Chrome, html_path: str):
    """
    Take a screenshot of the SVG in an HTML file and save it as PNG.

    Args:
        driver: Chrome WebDriver instance
        html_path: Absolute path to the HTML file
    """
    import os
    from selenium.common.exceptions import NoSuchElementException

    # Ensure absolute path is used
    html_path = os.path.abspath(html_path)

    # Generate output paths
    base_path = os.path.splitext(html_path)[0]
    full_screenshot_path = f"{base_path}_full.png"
    png_path = f"{base_path}.png"

    try:
        # Load HTML file
        driver.get(f'file://{html_path}')

        # Force browser background to transparent
        try:
            driver.execute_cdp_cmd('Emulation.setDefaultBackgroundColorOverride', {'color': {'r': 0, 'g': 0, 'b': 0, 'a': 0}})
        except Exception as e:
            print(f"Failed to set transparent background (CDP may not be supported): {e}")

        _dpr = 2

        time.sleep(0.3)  # Wait to ensure SVG is fully loaded

        # Find SVG element
        try:
            svg = driver.find_element("css selector", "svg")
        except NoSuchElementException:
            raise Exception(f"SVG element not found in HTML file: {html_path}")

        # Get SVG dimensions
        svg_width = driver.execute_script("return arguments[0].getBoundingClientRect().width;", svg)
        svg_height = driver.execute_script("return arguments[0].getBoundingClientRect().height;", svg)

        if svg_width <= 0 or svg_height <= 0:
            raise Exception(f"Invalid SVG dimensions: width={svg_width}, height={svg_height}")

        required_width = int(svg_width + 500)
        required_height = int(svg_height + 500)

        # Always set a sufficiently large window size to avoid truncation from the default 800x600
        # Ensure at least 1920x1080; use SVG dimensions if larger
        target_width = max(required_width, 800) #WARNING!!!!!
        target_height = max(required_height, 800)
        
        driver.set_window_size(target_width, target_height)

        # Set high-resolution screenshot (2x device pixel ratio)
        try:
            driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
                'width': target_width, 'height': target_height,
                'deviceScaleFactor': _dpr, 'mobile': False,
            })
        except Exception as e:
            print(f"Failed to set high resolution, falling back to 1x: {e}")
            _dpr = 1

        time.sleep(0.3)

        # Get SVG position and size (CSS pixels)
        location = svg.location
        size = svg.size

        x = location['x']
        y = location['y']
        width = size['width']
        height = size['height']

        driver.save_screenshot(full_screenshot_path)

        image = Image.open(full_screenshot_path)
        image = image.convert("RGBA")

        # Crop coordinates need to be multiplied by deviceScaleFactor
        left = round(x * _dpr)
        top = round(y * _dpr)
        right = round((x + width) * _dpr)
        bottom = round((y + height) * _dpr)

        img_width, img_height = image.size
        left = max(0, min(left, img_width))
        top = max(0, min(top, img_height))
        right = max(left, min(right, img_width))
        bottom = max(top, min(bottom, img_height))

        cropped_image = image.crop((left, top, right, bottom))
        cropped_image.save(png_path)

        # Clean up full screenshot file
        try:
            if os.path.exists(full_screenshot_path):
                os.remove(full_screenshot_path)
        except:
            pass

    except Exception as e:
        print(f"Screenshot failed: {e}")
        raise

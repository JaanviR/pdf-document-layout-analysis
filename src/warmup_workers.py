"""Send warmup requests to force GPU memory allocation in each Gunicorn worker.

Each worker lazy-loads CUDA memory on its first inference request. If multiple
unwarmed workers receive simultaneous requests, the concurrent CUDA allocation
can fail silently. This script sends sequential requests at startup so every
worker is fully initialized before real traffic arrives.
"""
import sys
import time
import urllib.request
import json


def wait_for_service(timeout=180):
    for i in range(1, timeout + 1):
        try:
            urllib.request.urlopen("http://127.0.0.1:5060/", timeout=2)
            print(f"[warmup] Service ready after {i}s")
            return True
        except Exception:
            time.sleep(1)
    return False


def create_warmup_pdf() -> bytes:
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Warmup page for GPU memory initialization.", fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def send_warmup(pdf_bytes: bytes, worker_num: int, total: int):
    boundary = "warmup_boundary_abc123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="warmup.pdf"\r\n'
        f"Content-Type: application/pdf\r\n\r\n"
    ).encode() + pdf_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        "http://127.0.0.1:5060/",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        result = json.loads(resp.read())
        print(f"[warmup] Worker {worker_num}/{total} ready ({len(result)} segments)")
    except Exception as e:
        print(f"[warmup] Worker {worker_num}/{total}: {e}")


def main():
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    print(f"[warmup] Waiting for service...")
    if not wait_for_service():
        print("[warmup] Service did not start in time, skipping warmup")
        return

    print(f"[warmup] Warming up {num_workers} workers...")

    pdf_bytes = create_warmup_pdf()
    for i in range(1, num_workers + 1):
        send_warmup(pdf_bytes, i, num_workers)

    print("[warmup] All workers warmed up.")


if __name__ == "__main__":
    main()

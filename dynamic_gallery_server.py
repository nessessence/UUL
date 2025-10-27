import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote
from natsort import natsorted

class GalleryHTTPRequestHandler(SimpleHTTPRequestHandler):
    # Helper to compute parent directory URL
    def _parent_href(self):
        p = self.path
        if not p.endswith('/'):
            p += '/'
        p = p.rstrip('/')
        if p == '' or p == '/':
            return None  # already at root
        parent = p.rsplit('/', 1)[0]
        return parent + '/' if parent else '/'

    def list_directory(self, path):
        try:
            file_list = os.listdir(path)
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None

        # Sort naturally
        file_list = natsorted(file_list)

        image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'))]
        if image_files:
            return self.generate_gallery(path, image_files)

        # Default: show as links (with natural sort)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        displaypath = unquote(self.path)
        self.wfile.write(f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Directory listing for {displaypath}</title>
<style>
    body {{ background-color: #111; color: #eee; font-family: sans-serif; }}
    a {{ color: #5fd7ff; text-decoration: none; }}
</style>
</head><body>
<h2>Directory listing for {displaypath}</h2><hr><ul>
""".encode("utf-8"))

        # ⬅️ Add "Go back" link at the top
        parent = self._parent_href()
        if parent:
            self.wfile.write(f'<li><a href="{parent}">⬅️ Go back</a></li>\n'.encode("utf-8"))

        # List files and directories
        for name in file_list:
            fullname = os.path.join(path, name)
            displayname = linkname = name
            if os.path.isdir(fullname):
                displayname = linkname = name + "/"
            self.wfile.write(f'<li><a href="{linkname}">{displayname}</a></li>\n'.encode("utf-8"))

        self.wfile.write(b"</ul><hr></body></html>")
        return None

    def generate_gallery(self, path, image_files):
        encoded_path = unquote(self.path)
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()

        self.wfile.write(b"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Image Gallery</title>
<style>
    body { background-color: #111; color: #eee; font-family: sans-serif; }
    .grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 10px; }
    .item { text-align: center; font-size: 12px; word-break: break-all; }
    img { width: 100%; height: auto; border-radius: 4px; }
</style></head><body>
""")

        # Header + Go Back link
        self.wfile.write(f"<h2>Gallery: {encoded_path}</h2>".encode("utf-8"))
        parent = self._parent_href()
        if parent:
            self.wfile.write(f'<p><a href="{parent}">⬅️ Go back</a></p>'.encode("utf-8"))

        self.wfile.write(b'<div class="grid">\n')

        # Display all images
        for fname in image_files:
            item = f"""<div class="item">
    <img src="{fname}" alt="{fname}">
    <div>{fname}</div>
</div>"""
            self.wfile.write(item.encode("utf-8"))

        self.wfile.write(b"</div></body></html>")
        return None


def run_server(port=8080, root_dir="."):
    os.chdir(root_dir)
    server = ThreadingHTTPServer(("", port), GalleryHTTPRequestHandler)
    print(f"Serving on http://localhost:{port}/")
    server.serve_forever()


if __name__ == "__main__":
    run_server(port=8080, root_dir=".")


# import os
# from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
# from urllib.parse import unquote
# from natsort import natsorted

# class GalleryHTTPRequestHandler(SimpleHTTPRequestHandler):
#     def list_directory(self, path):
#         try:
#             file_list = os.listdir(path)
#         except OSError:
#             self.send_error(404, "No permission to list directory")
#             return None

#         # Sort naturally
#         file_list = natsorted(file_list)

#         image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp','.gif'))]
#         if image_files:
#             return self.generate_gallery(path, image_files)

#         # Default: show as links (with natural sort)
#         self.send_response(200)
#         self.send_header("Content-type", "text/html; charset=utf-8")
#         self.end_headers()

#         displaypath = unquote(self.path)
#         self.wfile.write(f"""<!DOCTYPE html>
# <html><head>
# <meta charset="utf-8">
# <title>Directory listing for {displaypath}</title>
# <style>
#     body {{ background-color: #111; color: #eee; font-family: sans-serif; }}
#     a {{ color: #5fd7ff; text-decoration: none; }}
# </style>
# </head><body>
# <h2>Directory listing for {displaypath}</h2><hr><ul>
# """.encode("utf-8"))

#         for name in file_list:
#             fullname = os.path.join(path, name)
#             displayname = linkname = name
#             if os.path.isdir(fullname):
#                 displayname = linkname = name + "/"
#             self.wfile.write(f'<li><a href="{linkname}">{displayname}</a></li>\n'.encode("utf-8"))

#         self.wfile.write(b"</ul><hr></body></html>")
#         return None

#     def generate_gallery(self, path, image_files):
#         encoded_path = unquote(self.path)
#         self.send_response(200)
#         self.send_header("Content-type", "text/html; charset=utf-8")
#         self.end_headers()

#         self.wfile.write(b"""<!DOCTYPE html>
# <html><head><meta charset="UTF-8"><title>Image Gallery</title>
# <style>
#     body { background-color: #111; color: #eee; font-family: sans-serif; }
#     .grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 10px; }
#     .item { text-align: center; font-size: 12px; word-break: break-all; }
#     img { width: 100%; height: auto; border-radius: 4px; }
# </style></head><body>
# <h2>Gallery: """ + encoded_path.encode('utf-8') + b"""</h2><div class="grid">
# """)

#         for fname in image_files:
#             item = f"""<div class="item">
#     <img src="{fname}" alt="{fname}">
#     <div>{fname}</div>
# </div>"""
#             self.wfile.write(item.encode("utf-8"))

#         self.wfile.write(b"</div></body></html>")
#         return None

# def run_server(port=8080, root_dir="."):
#     os.chdir(root_dir)
#     server = ThreadingHTTPServer(("", port), GalleryHTTPRequestHandler)
#     print(f"Serving on http://localhost:{port}/")
#     server.serve_forever()

# if __name__ == "__main__":
#     run_server(port=8080, root_dir=".")

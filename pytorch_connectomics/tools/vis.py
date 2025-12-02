from tensorboard import program
import socket

def find_free_port(preferred=6006):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", preferred)) != 0:
            return preferred
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def launch_tensorboard(logdir="outputs/SNEMI_UNet"):
    port = find_free_port(6006)
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--port", str(port)])
    url = tb.launch()
    print(f"TensorBoard listening on {url}")
    return tb

if __name__ == "__main__":
    tb = launch_tensorboard()
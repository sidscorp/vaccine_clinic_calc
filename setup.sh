mkdir -p ~/.streamlit/
echo "[general]
email = \"siddharthanambiar@gmail.com\"
" > ~/.streamlit/credentials.toml
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml

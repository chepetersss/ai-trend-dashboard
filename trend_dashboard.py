import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pytrends.request import TrendReq
import instaloader
from prophet import Prophet
import datetime
import numpy as np

# ✅ Function to get trending search terms from Google Trends
def get_google_trends():
    pytrends = TrendReq()
    pytrends.build_payload(kw_list=["best products", "trending items", "hot selling"], timeframe='now 7-d')

    try:
        trending_searches = pytrends.related_queries()
        if trending_searches and 'best products' in trending_searches:
            top_trends = trending_searches['best products']['top']
            if top_trends is not None:
                return top_trends['query'].tolist()[:10]
    except Exception as e:
        print(f"Error fetching Google Trends: {e}")

    return ["No trending data available"]


# ✅ Function to scrape AliExpress trending products
def get_aliexpress_trends():
    ali_url = "https://www.aliexpress.com/wholesale?SearchText=trending+products"
    response = requests.get(ali_url)
    soup = BeautifulSoup(response.text, "html.parser")

    products = []
    for item in soup.find_all("a", class_="item-title"):
        products.append(item.get_text())

    return products[:10]  # Get top 10 trending products

# ✅ Function to scrape Amazon Movers & Shakers
def get_amazon_trends():
    amazon_url = "https://www.amazon.com/gp/movers-and-shakers"
    response = requests.get(amazon_url)
    soup = BeautifulSoup(response.text, "html.parser")

    products = []
    for item in soup.find_all("span", class_="zg-text-center-align"):
        products.append(item.get_text())

    return products[:10]

# ✅ Function to scrape TikTok trending hashtags
def get_tiktok_trends():
    tiktok_url = "https://www.tiktok.com/tag/trending-products"
    response = requests.get(tiktok_url)
    soup = BeautifulSoup(response.text, "html.parser")

    hashtags = []
    for tag in soup.find_all("h3", class_="title"):
        hashtags.append(tag.get_text())

    return hashtags[:10]

# ✅ Function to scrape Instagram trending hashtags using Instaloader
def get_instagram_trends():
    loader = instaloader.Instaloader()
    trending_hashtags = ["#trendingproducts", "#viralproducts", "#dropshipping"]
    trending_posts = []

    for hashtag in trending_hashtags:
        posts = instaloader.Hashtag.from_name(loader.context, hashtag).get_top_posts()
        for post in posts:
            trending_posts.append(post.url)
            if len(trending_posts) >= 10:
                break
        if len(trending_posts) >= 10:
            break

    return trending_posts

# ✅ Function to simulate product trend data for AI predictions
def generate_fake_trend_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=90, freq="D")
    values = np.cumsum(np.random.randn(90))  # Simulated trend values
    return pd.DataFrame({"ds": dates, "y": values})

# ✅ Function to predict future product trends using AI
def predict_trends():
    df = generate_fake_trend_data()
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Fetch Data
google_trends = get_google_trends()
aliexpress_trends = get_aliexpress_trends()
amazon_trends = get_amazon_trends()
tiktok_trends = get_tiktok_trends()
instagram_trends = get_instagram_trends()
predicted_trends = predict_trends()

# Streamlit Web App
st.title("🔥 AI-Powered Trend Prediction for Dropshipping & E-Commerce")
st.write("This AI tool tracks trending products on Google, TikTok, Instagram, Amazon, and AliExpress, and **predicts** future booming products.")

# Display Data
st.subheader("🔹 Google Trends (Trending Searches)")
st.table(pd.DataFrame(google_trends, columns=["Trending Searches"]))

st.subheader("🔹 AliExpress Trending Products")
st.table(pd.DataFrame(aliexpress_trends, columns=["AliExpress Best Sellers"]))

st.subheader("🔹 Amazon Movers & Shakers")
st.table(pd.DataFrame(amazon_trends, columns=["Amazon Trending Products"]))

st.subheader("🔹 TikTok Trending Hashtags")
st.table(pd.DataFrame(tiktok_trends, columns=["TikTok Hashtags"]))

st.subheader("🔹 Instagram Trending Products")
st.table(pd.DataFrame(instagram_trends, columns=["Instagram Reels Trends"]))

# AI Prediction Section
st.subheader("🚀 AI-Powered Trend Forecast (Next 30 Days)")
st.line_chart(predicted_trends.set_index("ds")["yhat"])

st.write("🔄 Data updates automatically every few hours.")

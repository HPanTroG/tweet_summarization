# import the module 
import tweepy 
  
# assign the values accordingly 
consumer_key = "krR21kgDqdhDvJizLbiv90bDP" 
consumer_secret = "ERTZWDCEoSSjCPSdJ7Jl3cNB6Mqi7reycyQBFDFfVONsyIl44k" 
access_token = "1323575376669515776-PyuwlRrOj8P703JwYBsdawusGFVo71" 
access_token_secret = "WUjbdAZeEJHT6Bd4FSjmtywwwejlBCx5z3QGzoFT78MO1" 
  
# authorization of consumer key and consumer secret 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
  
# set access to user's access key and access secret  
auth.set_access_token(access_token, access_token_secret) 
  
# calling the api  
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True) 
  
# the ID of the status 
id = 356249115857920000
  
# fetching the status 
status = api.get_status(id) 
  
# fetching the lang attribute 
lang = status.lang  
  
print("The language of the status is : " + lang) 

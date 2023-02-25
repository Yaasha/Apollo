![docker build](https://github.com/Yaasha/Apollo/actions/workflows/docker-publish.yml/badge.svg)

# Apollo

This project implements a trading bot designed to utilize deep reinforcement learning to trade stocks from the DOW 30 stocks index through the Alpaca broker API.

The bot includes an autorun mode that can be deployed in a Docker container, which automatically trains the model every day prior to the stock market opening on data from the past two weeks. Assuming the model is profitable during evaluation, it is then used to suggest trades every minute while the stock market is open. These trades are executed through the Alpaca paper trading API. Optionally, the bot also sends a Discord message with the daily trading performance metrics at the end of each trading day.


## Disclaimer
Please note that this trading bot is designed for paper trading and experimental purposes only. The use of this bot with real money is at your own risk. There is no guarantee that the deep reinforcement learning model used in this project will be profitable, and past performance does not guarantee future results.

## Selfhosting

If you want to self-host your own apollo instance for testing and experimentation, you can do so using the docker image.

### Docker run

```
 docker run \
  -e "API_KEY=<KEY>" \
  -e "API_SECRET=<SECRET>" \
  -e "API_BASE_URL=https://paper-api.alpaca.markets" \
  -e "DATA_URL=wss://data.alpaca.markets" \
  -e "DISCORD_WEBHOOK_URL=<link to discord channel webhook>" \
  ghcr.io/yaasha/apollo
```

### Docker compose
```
services:
  apollo:
    image: ghcr.io/yaasha/apollo
    container_name: apollo
    restart: unless_stopped
    environment:
      - API_KEY=KEY # your Alpaca paper trading account API key
      - API_SECRET=SECRET # your Alpaca paper trading account API secret
      - API_BASE_URL=https://paper-api.alpaca.markets
      - DATA_URL=wss://data.alpaca.markets
      - DISCORD_WEBHOOK_URL=link to discord channel webhook # optional
```


## Credits
This project uses the FinRL library from the AI4Finance foundation (https://github.com/AI4Finance-Foundation/FinRL) for the DRL model implementaion and the code is based on the tutorial notebooks from https://github.com/AI4Finance-Foundation/FinRL-Tutorials.

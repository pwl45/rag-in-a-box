import requests
import pandas as pd
import subprocess
from time import sleep
import os

class EdgarClient:
    def __init__(self):
        self.header = {
            "User-Agent": "MyUserAgent/1.0"
        }

    def get_cik_from_ticker(self, ticker):
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers=self.header)
        data = response.json()

        for company in data.values():
            if company["ticker"] == ticker:
                return company["cik_str"]

        return None

    def get_company_filings(self, cik):
        url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
        response = requests.get(url, headers=self.header)
        data = response.json()
        return data

    def get_company_filings_from_ticker(self, ticker):
        cik = self.get_cik_from_ticker(ticker)
        return self.get_company_filings(cik)

    def get_recent_filing(self, ticker, filing_type='any'):
        cik = self.get_cik_from_ticker(ticker)
        if cik is None:
            raise ValueError(f"No CIK found for ticker {ticker}")

        company_filings = self.get_company_filings(cik)
        print(company_filings)
        filings_df = pd.DataFrame(company_filings["filings"]["recent"])
        if filing_type == 'any':
            filings_df = filings_df[filings_df.form.isin(['10-Q','10-K'])].sort_values("filingDate", ascending=False)
        else:
            filings_df = filings_df[filings_df.form == filing_type].sort_values("filingDate", ascending=False)

        print(filings_df)
        if len(filings_df) == 0:
            raise ValueError(f"No {filing_type} filings found for {ticker}")

        latest_filing = filings_df.iloc[0]
        accession_number = latest_filing.accessionNumber.replace("-", "")
        file_name = latest_filing.primaryDocument
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{file_name}"
        # url='https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm'

        user_agent = "MyUserAgent/1.0"
        # Prepare the curl command
        curl_command = f'curl -H "User-Agent: {user_agent}" {url}'

        # response = requests.get(url, headers=self.header)

        result = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return latest_filing, result.stdout

    def get_recent_filings(self, ticker, filing_type, n=3):
        cik = self.get_cik_from_ticker(ticker)
        if cik is None:
            raise ValueError(f"No CIK found for ticker {ticker}")

        company_filings = self.get_company_filings(cik)
        filings_df = pd.DataFrame(company_filings["filings"]["recent"])
        if filing_type == 'any':
            filings_df = filings_df[filings_df.form.isin(['10-Q', '10-K'])].sort_values("filingDate", ascending=False)
        else:
            filings_df = filings_df[filings_df.form == filing_type].sort_values("filingDate", ascending=False)

        filings = []
        for _, row in filings_df.iterrows():
            if len(filings) < n:
                accession_number = row.accessionNumber.replace("-", "")
                file_name = row.primaryDocument
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{file_name}"
                user_agent = "MyUserAgent/1.0"
                curl_command = f'curl -H "User-Agent: {user_agent}" {url}'
                sleep(1)
                result = subprocess.run(curl_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                filings.append((row, result.stdout))
            else:
                break
        return filings

client = EdgarClient()
# exit(0)
# tickers = ['ROIV', 'PFE', 'ARQT', 'MRK', 'IMVT', 'IRWD', 'SAGE', 'LLY']
tickers = ['ROIV', 'PFE', 'ARQT', 'MRK', 'IMVT', 'BBIO', 'REGN']
# tickers = ['RHHBY','ARGX']
# map tickers to their company names
ticker_name_map = {
    'ROIV': 'roivant',
    'PFE': 'pfizer',
    'ARQT': 'arcutis',
    'MRK': 'merck',
    'IMVT': 'immunovant',
    'BBIO': 'bridgebio',
    'REGN': 'regeneron',
}
# tickers = ['ARGX']
filing_type_map = {
    '10-K': 'annual',
    '10-Q': 'quarterly',
    '20-F': 'quartlery',
}

file_name = None
filing_content = None
for ticker in tickers:
    company_name = ticker_name_map[ticker]
    for filing_type in ['10-K','10-Q']:
        results = client.get_recent_filings(ticker, filing_type, n=5)
        filing_type_name = filing_type_map[filing_type]
        for file_metadata, filing_content in results:
            filing_date = file_metadata.filingDate
            filename = f'output/{filing_type_name}-financial-reports/{company_name}/{filing_type}-{filing_date}.html'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                f.write(filing_content)
            print(filing_content[:1000])
            print(filename)
            sleep(1)

# file_name = None
# filing_content = None
# for ticker in tickers:
#     company_name = ticker_name_map[ticker]
#     for filing_type in ['10-K','10-Q']:
#         filing_type_name = filing_type_map[filing_type]
#         file_metadata, filing_content = client.get_recent_filing(ticker, filing_type)
#         filing_date = file_metadata.filingDate
#         filename = f'output/{filing_type_name}-financial-reports/{company_name}/{filing_type}-{filing_date}.html'
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#         with open(filename, 'w') as f:
#             f.write(filing_content)
#         print(filing_content[:1000])
#         print(filename)
#         sleep(1)
#     # write filing to output /

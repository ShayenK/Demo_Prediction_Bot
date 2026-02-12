import os
from web3 import Web3
from web3.constants import MAX_INT
from web3.middleware import ExtraDataToPOAMiddleware  # Updated for web3.py v6+
from dotenv import load_dotenv
load_dotenv()

rpc_url = "https://polygon-rpc.com"
priv_key = os.getenv('PRIVATE_KEY')
pub_key = os.getenv('PUBLIC_KEY')
chain_id = 137
erc20_approve_abi = '[{"constant": false,"inputs": [{"name": "_spender","type": "address" },{ "name": "_value", "type": "uint256" }],"name": "approve","outputs": [{ "name": "", "type": "bool" }],"payable": false,"stateMutability": "nonpayable","type": "function"}]'
erc1155_set_approval_abi = '[{"inputs": [{ "internalType": "address", "name": "operator", "type": "address" },{ "internalType": "bool", "name": "approved", "type": "bool" }],"name": "setApprovalForAll","outputs": [],"stateMutability": "nonpayable","type": "function"}]'
usdc_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
ctf_address = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
web3 = Web3(Web3.HTTPProvider(rpc_url))
web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
# === KEY FIX: Use pending nonce to handle stuck txs ===
nonce = web3.eth.get_transaction_count(pub_key, 'pending')
usdc = web3.eth.contract(address=usdc_address, abi=erc20_approve_abi)
ctf = web3.eth.contract(address=ctf_address, abi=erc1155_set_approval_abi)
def get_gas_price():
    """Get gas price with 20% buffer for reliability"""
    base_price = web3.eth.gas_price
    return int(base_price * 1.2)  # 20% higher
def send_and_wait(raw_tx):
    signed_tx = web3.eth.account.sign_transaction(raw_tx, private_key=priv_key)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    print(f"Success: {receipt.transactionHash.hex()} (status: {receipt.status})")
    return receipt
print(f"Starting Process... (nonce: {nonce})")

# === CTF Exchange ===
# USDC approve
raw_tx = usdc.functions.approve(
    "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E", int(MAX_INT, 0)
).build_transaction({
    "chainId": chain_id,
    "from": pub_key,
    "nonce": nonce,
    "gasPrice": web3.eth.gas_price  # Auto gas price
})
send_and_wait(raw_tx)
nonce += 1

# CTF setApprovalForAll
raw_tx = ctf.functions.setApprovalForAll("0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E", True).build_transaction({
    "chainId": chain_id,
    "from": pub_key,
    "nonce": nonce,
    "gasPrice": web3.eth.gas_price
})
send_and_wait(raw_tx)
nonce += 1

# === Neg Risk CTF Exchange ===
# USDC approve
raw_tx = usdc.functions.approve(
    "0xC5d563A36AE78145C45a50134d48A1215220f80a", int(MAX_INT, 0)
).build_transaction({
    "chainId": chain_id,
    "from": pub_key,
    "nonce": nonce,
    "gasPrice": web3.eth.gas_price
})
send_and_wait(raw_tx)
nonce += 1

# CTF setApprovalForAll
raw_tx = ctf.functions.setApprovalForAll("0xC5d563A36AE78145C45a50134d48A1215220f80a", True).build_transaction({
    "chainId": chain_id,
    "from": pub_key,
    "nonce": nonce,
    "gasPrice": web3.eth.gas_price
})
send_and_wait(raw_tx)
nonce += 1

# === Neg Risk Adapter ===
# USDC approve
raw_tx = usdc.functions.approve(
    "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296", int(MAX_INT, 0)
).build_transaction({
    "chainId": chain_id,
    "from": pub_key,
    "nonce": nonce,
    "gasPrice": web3.eth.gas_price
})
send_and_wait(raw_tx)
nonce += 1

# CTF setApprovalForAll
raw_tx = ctf.functions.setApprovalForAll("0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296", True).build_transaction({
    "chainId": chain_id,
    "from": pub_key,
    "nonce": nonce,
    "gasPrice": web3.eth.gas_price
})
send_and_wait(raw_tx)

print("All 6 approvals completed! Your wallet is now permanently set up for Polymarket trading.")
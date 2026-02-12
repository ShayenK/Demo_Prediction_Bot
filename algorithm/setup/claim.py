import os
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from dotenv import load_dotenv
from config import (
    RPC_URL,
    PUBLIC_KEY,
    PRIVATE_KEY
)
load_dotenv()

USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
NEG_RISK_CTF_ADDRESS = Web3.to_checksum_address("0xC5d563A36AE78145C45a50134d48A1215220f80a")

# Add approval ABI for CTF contract
CTF_APPROVAL_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "operator", "type": "address"},
            {"internalType": "bool", "name": "approved", "type": "bool"}
        ],
        "name": "setApprovalForAll",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "address", "name": "operator", "type": "address"}
        ],
        "name": "isApprovedForAll",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Existing ABIs (corrected resolution ABI from earlier)
RESOLUTION_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256", "name": "index", "type": "uint256"}
        ],
        "name": "payoutNumerators", 
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

CTF_BALANCE_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "account", "type": "address"},
            {"internalType": "uint256", "name": "id", "type": "uint256"}
        ],
        "name": "balanceOf",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

CTF_REDEEM_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32", "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

NEG_RISK_CTF_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "conditionId", "type": "bytes32"},
            {"internalType": "uint256[]", "name": "amounts", "type": "uint256[]"}
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


def get_web3() -> Web3:
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3

def _to_bytes32(hex_string: str) -> bytes:
    clean = hex_string[2:] if hex_string.startswith("0x") else hex_string
    return bytes.fromhex(clean.zfill(64))

def get_token_balance(token_id: str) -> int:
    try:
        w3 = get_web3()
        ctf_contract = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_BALANCE_ABI)
        balance = ctf_contract.functions.balanceOf(PUBLIC_KEY, int(token_id)).call()
        return balance
    except Exception as e:
        print(f"ERROR: balance query failed for token {token_id}: {e}")
        return 0

def get_market_resolution(condition_id: str) -> list:
    """
    Get market resolution using corrected ABI
    """
    try:
        w3 = get_web3()
        condition_bytes = _to_bytes32(condition_id)
        ctf_contract = w3.eth.contract(address=CTF_ADDRESS, abi=RESOLUTION_ABI)
        
        payout_numerators = []
        
        # Get YES outcome (index 0)
        try:
            yes_payout = ctf_contract.functions.payoutNumerators(condition_bytes, 0).call()
            payout_numerators.append(yes_payout)
        except Exception as e:
            print(f"ERROR: Could not get YES payout: {e}")
            return []
        
        # Get NO outcome (index 1)  
        try:
            no_payout = ctf_contract.functions.payoutNumerators(condition_bytes, 1).call()
            payout_numerators.append(no_payout)
        except Exception as e:
            print(f"ERROR: Could not get NO payout: {e}")
            return []
            
        return payout_numerators
        
    except Exception as e:
        print(f"ERROR: could not get resolution for {condition_id}: {e}")
        return []

def check_and_set_approval(operator_address: str) -> bool:
    """
    Check if NegRisk adapter is approved, and approve if not
    """
    try:
        w3 = get_web3()
        ctf_contract = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_APPROVAL_ABI)
        
        # Check current approval
        is_approved = ctf_contract.functions.isApprovedForAll(PUBLIC_KEY, operator_address).call()
        
        if is_approved:
            print(f"INFO: ✅ {operator_address[:10]}... already approved")
            return True
        
        print(f"INFO: 🔄 Setting approval for {operator_address[:10]}...")
        
        # Set approval
        gas_params = {
            'from': PUBLIC_KEY,
            'nonce': w3.eth.get_transaction_count(PUBLIC_KEY),
            'gas': 100000,  # Standard gas for approval
            'maxFeePerGas': w3.eth.get_block('latest')['baseFeePerGas'] * 2,
            'maxPriorityFeePerGas': w3.to_wei(30, 'gwei')
        }
        
        tx = ctf_contract.functions.setApprovalForAll(
            operator_address, 
            True
        ).build_transaction(gas_params)
        
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        print(f"INFO: Approval tx sent: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        if receipt['status'] == 1:
            print(f"INFO: ✅ Approval successful!")
            return True
        else:
            print(f"ERROR: Approval failed")
            return False
            
    except Exception as e:
        print(f"ERROR: Failed to set approval: {e}")
        return False

def calculate_redeemable_amounts(yes_balance: int, no_balance: int, payout_numerators: list) -> tuple:
    if not payout_numerators or len(payout_numerators) != 2:
        return 0, 0
    yes_payout_ratio = payout_numerators[0]
    no_payout_ratio = payout_numerators[1]
    total_payout = sum(payout_numerators)
    
    if total_payout == 0:
        return 0, 0
    yes_redeemable = (yes_balance * yes_payout_ratio) // total_payout
    no_redeemable = (no_balance * no_payout_ratio) // total_payout
    return yes_redeemable, no_redeemable

def redeem_position(
    condition_id: str,
    yes_token_id: str,
    no_token_id: str,
    is_neg_risk: bool = False
) -> int | None:
    try:
        w3 = get_web3()
        yes_balance = get_token_balance(yes_token_id)
        no_balance = get_token_balance(no_token_id)
        print(f"INFO: condition={condition_id[:10]}..., yes_bal={yes_balance}, no_bal={no_balance}, neg_risk={is_neg_risk}")
        
        if yes_balance == 0 and no_balance == 0:
            print(f"INFO: No balance to redeem for {condition_id[:10]}...")
            return 0
            
        payout_numerators = get_market_resolution(condition_id)
        if not payout_numerators:
            print(f"WARN: Market {condition_id[:10]}... may not be resolved yet")
            return None
            
        print(f"INFO: Market resolution: {payout_numerators}")
        
        # Calculate expected redemption amount
        yes_redeemable, no_redeemable = calculate_redeemable_amounts(
            yes_balance, no_balance, payout_numerators
        )
        total_redeemable = yes_redeemable + no_redeemable
        
        if total_redeemable == 0:
            print(f"INFO: No redeemable value for {condition_id[:10]}...")
            return 0
        
        # CRITICAL FIX: Set approval for NegRisk if needed
        if is_neg_risk:
            approval_success = check_and_set_approval(NEG_RISK_CTF_ADDRESS)
            if not approval_success:
                print(f"ERROR: Failed to set approval for NegRisk adapter")
                return None
            
        condition_bytes = _to_bytes32(condition_id)
        
        # Gas estimation and transaction building
        try:
            if is_neg_risk:
                amounts = [yes_balance, no_balance]
                neg_risk_adapter = w3.eth.contract(address=NEG_RISK_CTF_ADDRESS, abi=NEG_RISK_CTF_ABI)
                gas_estimate = neg_risk_adapter.functions.redeemPositions(
                    condition_bytes,
                    amounts
                ).estimate_gas({'from': PUBLIC_KEY})
            else:
                index_sets = []
                if yes_redeemable > 0:
                    index_sets.append(1)
                if no_redeemable > 0:
                    index_sets.append(2)
                if not index_sets:
                    return 0
                ctf_contract = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_REDEEM_ABI)
                gas_estimate = ctf_contract.functions.redeemPositions(
                    USDC_ADDRESS,
                    bytes(32),
                    condition_bytes,
                    index_sets
                ).estimate_gas({'from': PUBLIC_KEY})
        except Exception as e:
            print(f"WARN: Gas estimation failed, using default: {e}")
            gas_estimate = 350000
            
        gas_limit = int(gas_estimate * 1.2)
        base_fee = w3.eth.get_block('latest')['baseFeePerGas']
        priority_fee = max(w3.eth.max_priority_fee, w3.to_wei(30, 'gwei'))
        max_fee = base_fee * 2 + priority_fee
        
        gas_params = {
            'from': PUBLIC_KEY,
            'nonce': w3.eth.get_transaction_count(PUBLIC_KEY),
            'gas': gas_limit,
            'maxFeePerGas': max_fee,
            'maxPriorityFeePerGas': priority_fee,
        }
        
        # Build transaction
        if is_neg_risk:
            amounts = [yes_balance, no_balance]
            neg_risk_adapter = w3.eth.contract(address=NEG_RISK_CTF_ADDRESS, abi=NEG_RISK_CTF_ABI)
            tx = neg_risk_adapter.functions.redeemPositions(
                condition_bytes,
                amounts
            ).build_transaction(gas_params)
        else:
            index_sets = []
            if yes_redeemable > 0:
                index_sets.append(1)
            if no_redeemable > 0:
                index_sets.append(2)
            if not index_sets:
                return 0
            ctf_contract = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_REDEEM_ABI)
            tx = ctf_contract.functions.redeemPositions(
                USDC_ADDRESS,
                bytes(32),
                condition_bytes,
                index_sets
            ).build_transaction(gas_params)
            
        print(f"INFO: Attempting redemption with gas limit: {gas_limit}")
        print(f"INFO: Expected to receive: {total_redeemable / 1e6:.6f} USDC")
        
        signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        print(f"INFO: tx sent: {tx_hash.hex()}")
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        if receipt['status'] == 1:
            print(f"INFO: ✅ redemption successful! tx: {tx_hash.hex()}")
            print(f"INFO: Gas used: {receipt['gasUsed']} / {gas_limit}")
            print(f"INFO: 💰 You should have received ~{total_redeemable / 1e6:.6f} USDC")
            return total_redeemable
        else:
            print(f"WARN: redemption tx reverted: {tx_hash.hex()}")
            print(f"INFO: Gas used: {receipt['gasUsed']} / {gas_limit}")
            return None
            
    except Exception as e:
        print(f"ERROR: redeem_position failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def is_market_resolved(condition_id: str) -> bool:
    payout_numerators = get_market_resolution(condition_id)
    return len(payout_numerators) > 0 and any(p > 0 for p in payout_numerators)

def check_position_status(condition_id: str, yes_token_id: str, no_token_id: str) -> dict:
    yes_balance = get_token_balance(yes_token_id)
    no_balance = get_token_balance(no_token_id)
    payout_numerators = get_market_resolution(condition_id)
    resolved = len(payout_numerators) > 0 and any(p > 0 for p in payout_numerators)
    yes_redeemable, no_redeemable = 0, 0
    if resolved:
        yes_redeemable, no_redeemable = calculate_redeemable_amounts(
            yes_balance, no_balance, payout_numerators
        )
    return {
        'condition_id': condition_id,
        'yes_balance': yes_balance,
        'no_balance': no_balance,
        'total_balance': yes_balance + no_balance,
        'resolved': resolved,
        'payout_numerators': payout_numerators,
        'yes_redeemable': yes_redeemable,
        'no_redeemable': no_redeemable,
        'total_redeemable': yes_redeemable + no_redeemable
    }
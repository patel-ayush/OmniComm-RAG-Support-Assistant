"""
Custom RAG Performance Test Suite
==================================
Tests the OmniComm RAG system against hand-crafted edge-case queries covering:
  - Direct lookups
  - Multi-hop reasoning (combining multiple KB articles)
  - Deprecated document handling
  - Policy enforcement (no workarounds)
  - Abstention (out-of-scope questions)
  - Temporal reasoning (old vs new data)
  - Adversarial queries (trying to trick the model)
"""
import sys
import os
import json
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_chain import get_rag_chain

# ─────────────────────────────────────────────────────────────
# Custom test queries with expected behavior assertions
# ─────────────────────────────────────────────────────────────
CUSTOM_TESTS = [
    # ── 1. Direct Lookup ──
    {
        "id": "custom_001",
        "question": "What is the Basic plan rate for calling Singapore?",
        "category": "direct_lookup",
        "expected_sources": ["kb_002"],
        "must_contain": ["0.05"],
        "must_not_contain": ["0.04"],  # 0.04 is the Pro rate, not Basic
        "should_abstain": False,
    },
    # ── 2. Multi-Hop: Combine pricing + roaming ──
    {
        "id": "custom_002",
        "question": "I'm on the Basic plan in Hong Kong. If I call the Netherlands, will I be charged more because I'm abroad?",
        "category": "multi_hop",
        "expected_sources": ["kb_002", "kb_004"],
        "must_contain": ["0.06"],  # NL Basic rate
        "must_not_contain": [],
        "should_abstain": False,
    },
    # ── 3. Deprecated Document Handling ──
    {
        "id": "custom_003",
        "question": "What was the old 2025 Basic rate for Indonesia before the price change?",
        "category": "deprecated_handling",
        "expected_sources": ["kb_019"],
        "must_contain": ["0.09"],  # Old deprecated rate
        "must_not_contain": [],
        "should_abstain": False,
    },
    # ── 4. Deprecated vs Active Conflict ──
    {
        "id": "custom_004",
        "question": "Is the current API rate limit 60 or 100 requests per second?",
        "category": "deprecated_vs_active",
        "expected_sources": ["kb_011"],
        "must_contain": ["100"],
        "must_not_contain": [],
        "should_abstain": False,
    },
    # ── 5. Policy Enforcement ──
    {
        "id": "custom_005",
        "question": "I need to call 911 through the platform. Can you help me set that up?",
        "category": "policy_enforcement",
        "expected_sources": ["kb_005"],
        "must_contain": ["not support", "emergency"],
        "must_not_contain": ["workaround", "try"],
        "should_abstain": False,
    },
    # ── 6. Abstention: Out-of-Scope ──
    {
        "id": "custom_006",
        "question": "Can I use OmniComm to send faxes to Japan?",
        "category": "abstention",
        "expected_sources": [],
        "must_contain": [],  # Should say "I don't have information"
        "must_not_contain": ["yes", "fax support"],
        "should_abstain": True,
    },
    # ── 7. Multi-Hop: Troubleshooting + Firewall ──
    {
        "id": "custom_007",
        "question": "Our calls disconnect after exactly 2 minutes and we use a corporate firewall. What ports and timeout settings should we check?",
        "category": "multi_hop_troubleshooting",
        "expected_sources": ["kb_013", "kb_014"],
        "must_contain": ["180", "UDP"],  # Firewall timeout + protocol
        "must_not_contain": [],
        "should_abstain": False,
    },
    # ── 8. Multi-Hop: Billing + Refund Timeline ──
    {
        "id": "custom_008",
        "question": "I was double-charged. What do I need to provide, and how long will the refund take?",
        "category": "multi_hop_billing",
        "expected_sources": ["kb_006", "kb_007"],
        "must_contain": ["call_id", "5 to 7"],
        "must_not_contain": [],
        "should_abstain": False,
    },
    # ── 9. Enterprise Pricing Abstention ──
    {
        "id": "custom_009",
        "question": "I'm an Enterprise customer. What exactly is my per-minute rate for Indonesia?",
        "category": "enterprise_abstention",
        "expected_sources": ["kb_003"],
        "must_contain": ["contract"],  # Should say it depends on contract
        "must_not_contain": [],
        "should_abstain": False,  # Not full abstention, but partial
    },
    # ── 10. Adversarial: Trying to extract hallucinated info ──
    {
        "id": "custom_010",
        "question": "What is OmniComm's office address in Singapore and who is the CEO?",
        "category": "adversarial_hallucination",
        "expected_sources": [],
        "must_contain": [],   # Should abstain
        "must_not_contain": [],
        "should_abstain": True,
    },
    # ── 11. Support SLA Multi-Hop ──
    {
        "id": "custom_011",
        "question": "If I'm on the Pro plan and have a P1 outage, what is my expected response time?",
        "category": "multi_hop_sla",
        "expected_sources": ["kb_009"],
        "must_contain": ["4 business hours"],  # Pro gets 4h, not 30min
        "must_not_contain": ["30 minutes"],  # 30min is Enterprise only
        "should_abstain": False,
    },
    # ── 12. Complex Temporal: Suspension + Payment ──
    {
        "id": "custom_012",
        "question": "We were flagged for suspicious activity and paid immediately. When will we be back online?",
        "category": "temporal_account",
        "expected_sources": ["kb_008"],
        "must_contain": ["24 hours", "manual review"],
        "must_not_contain": ["2 hours"],  # 2h is for normal card payment
        "should_abstain": False,
    },
]

def run_custom_tests():
    print("=" * 70)
    print("     OMNICOMM RAG CUSTOM PERFORMANCE TEST SUITE")
    print("=" * 70)
    
    rag_chain = get_rag_chain()
    
    passed = 0
    failed = 0
    results = []

    for test in CUSTOM_TESTS:
        print(f"\n{'─' * 60}")
        print(f"[{test['id']}] {test['category'].upper()}")
        print(f"Q: {test['question']}")
        
        try:
            result = rag_chain.invoke(test["question"])
            result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
            answer = result_dict.get("answer", "").lower()
            sources = result_dict.get("sources", [])
            
            print(f"A: {result_dict.get('answer', '')[:200]}...")
            print(f"Sources: {sources}")
            
            issues = []
            
            # Check must_contain
            for keyword in test["must_contain"]:
                if keyword.lower() not in answer:
                    issues.append(f"❌ Missing expected keyword: '{keyword}'")
            
            # Check must_not_contain
            for keyword in test["must_not_contain"]:
                if keyword.lower() in answer:
                    issues.append(f"❌ Contains forbidden keyword: '{keyword}'")
            
            # Check expected sources
            for src in test["expected_sources"]:
                if src not in sources:
                    # Also check inline citations
                    inline = re.findall(r"\[(kb_\d{3})\]", answer)
                    if src not in inline:
                        issues.append(f"⚠️  Missing expected source: {src}")
            
            # Check abstention
            if test["should_abstain"]:
                abstain_phrases = ["don't have", "do not have", "not available", "no information", "cannot", "not in"]
                if not any(phrase in answer for phrase in abstain_phrases):
                    issues.append("❌ Should have abstained but gave a non-abstaining answer")
            
            if issues:
                failed += 1
                status = "FAIL"
                for issue in issues:
                    print(f"  {issue}")
            else:
                passed += 1
                status = "PASS"
                print("  ✅ All assertions passed")
            
            results.append({
                "id": test["id"],
                "category": test["category"],
                "status": status,
                "issues": issues,
                "answer_preview": result_dict.get("answer", "")[:150],
                "sources": sources,
            })
            
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR: {e}")
            results.append({
                "id": test["id"],
                "category": test["category"],
                "status": "ERROR",
                "issues": [str(e)],
                "answer_preview": "",
                "sources": [],
            })

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"     RESULTS: {passed} PASSED | {failed} FAILED | {len(CUSTOM_TESTS)} TOTAL")
    print(f"     PASS RATE: {passed/len(CUSTOM_TESTS)*100:.0f}%")
    print(f"{'=' * 70}")
    
    # Save results to JSON
    with open("custom_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to custom_test_results.json")

if __name__ == "__main__":
    run_custom_tests()

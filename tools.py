"""
Tool functions for the customer support agent.

Every function returns a dict. Errors are returned as {"error": "..."} — never raised.
Every side-effect function writes to SQLite for audit.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic

from db import get_db, init_db
from rag import get_return_policy as _rag_get_return_policy

ORDERS_FILE = Path(__file__).parent / "data" / "orders.json"

# Ensure DB exists at import time
init_db()

# One Anthropic client shared across sentiment calls
_anthropic_client = Anthropic()


# ==================== Data loading ====================

def _load_orders() -> list[dict]:
    """Read orders.json fresh every call — mock data source, no caching for now."""
    with open(ORDERS_FILE) as f:
        return json.load(f)


def _find_order(order_id: str) -> dict | None:
    """Internal helper — returns the order dict or None."""
    return next((o for o in _load_orders() if o['order_id'] == order_id), None)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ==================== Read-only tools ====================

def lookup_order(order_id: str) -> dict:
    """Return an order by ID."""
    order = _find_order(order_id)
    if not order:
        return {"error": f"No order found with ID '{order_id}'."}
    return order


def lookup_customer_orders(customer_id: str) -> dict:
    """Return all orders for a customer, sorted by most recent first."""
    orders = [o for o in _load_orders() if o['customer_id'] == customer_id]
    orders.sort(key=lambda o: o['order_received_datetime'], reverse=True)
    if not orders:
        return {"error": f"No orders found for customer '{customer_id}'."}
    return {"customer_id": customer_id, "count": len(orders), "orders": orders}


def check_shipping(order_id: str) -> dict:
    """Mock shipping status. Returns tracking info based on order status."""
    order = _find_order(order_id)
    if not order:
        return {"error": f"No order found with ID '{order_id}'."}

    status = order['order_status']

    if status == 'placed':
        return {
            "order_id": order_id,
            "status": "placed",
            "tracking_available": False,
            "message": "Order received. Being prepared for shipment. No tracking yet.",
        }
    if status == 'in-progress':
        return {
            "order_id": order_id,
            "status": "in-progress",
            "tracking_available": False,
            "message": "Order packed and awaiting carrier pickup.",
        }
    if status == 'in-flight':
        return {
            "order_id": order_id,
            "status": "in-flight",
            "tracking_available": True,
            "tracking_number": f"USPS-{order_id[-6:]}",
            "current_location": "Regional sorting facility",
            "estimated_delivery": order['estimated_delivery_date'],
            "message": "Order in transit.",
        }
    if status == 'delivered':
        return {
            "order_id": order_id,
            "status": "delivered",
            "tracking_available": True,
            "delivered_on": order['order_delivered_date'],
            "message": f"Delivered on {order['order_delivered_date']}.",
        }
    if status == 'lost':
        return {
            "order_id": order_id,
            "status": "lost",
            "tracking_available": True,
            "message": "Marked lost by carrier. Customer eligible for refund or replacement.",
        }

    return {"error": f"Unknown order status: {status}"}


def get_return_policy(query: str) -> dict:
    """Wrap the RAG function so it fits the standard tool return shape."""
    results = _rag_get_return_policy(query, top_k=3)
    return {"query": query, "results": results}


# ==================== State-changing tools (write to DB) ====================

def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """Process a refund. IMPORTANT: Agent should not call this for amounts >= $50, perishables, or past-30-day orders."""
    order = _find_order(order_id)
    if not order:
        return {"error": f"No order found with ID '{order_id}'."}

    if amount <= 0:
        return {"error": "Refund amount must be positive."}
    if amount > order['total_cost']:
        return {"error": f"Refund amount ${amount:.2f} exceeds order total ${order['total_cost']:.2f}."}

    refund_id = _new_id("REF")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO refunds (refund_id, order_id, customer_id, amount, reason, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'processed', ?)""",
            (refund_id, order_id, order['customer_id'], amount, reason, datetime.now().isoformat()),
        )

    return {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount": amount,
        "status": "processed",
        "estimated_arrival": "5-10 business days to your original payment method",
        "message": f"Refund of ${amount:.2f} processed. Confirmation ID: {refund_id}.",
    }


def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel an order. Only works for orders in 'placed' or 'in-progress' status."""
    order = _find_order(order_id)
    if not order:
        return {"error": f"No order found with ID '{order_id}'."}

    if order['order_status'] not in ('placed', 'in-progress'):
        return {
            "error": f"Cannot cancel — order is in '{order['order_status']}' status. "
                     f"Only 'placed' or 'in-progress' orders can be cancelled directly."
        }

    with get_db() as conn:
        conn.execute(
            """INSERT INTO cancellations (order_id, customer_id, reason, cancelled_at)
               VALUES (?, ?, ?, ?)""",
            (order_id, order['customer_id'], reason, datetime.now().isoformat()),
        )

    return {
        "order_id": order_id,
        "status": "cancelled",
        "refund_amount": order['total_cost'],
        "message": f"Order {order_id} cancelled. Full refund of ${order['total_cost']:.2f} initiated to original payment method.",
    }


def create_ticket(customer_id: str, subject: str, category: str, details: str,
                  order_id: str = None) -> dict:
    """Create a support ticket to log the interaction. Call at end of every conversation."""
    ticket_id = _new_id("TKT")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO tickets (ticket_id, customer_id, order_id, subject, category, details, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'open', ?)""",
            (ticket_id, customer_id, order_id, subject, category, details, datetime.now().isoformat()),
        )
    return {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "status": "open",
        "message": f"Ticket {ticket_id} created.",
    }


def escalate_to_human(customer_id: str, reason: str, order_id: str = None,
                      priority: str = "medium") -> dict:
    """Escalate to human support. Use for: refund >= $50, perishables, past-30-day orders, human requested, sentiment negative."""
    if priority not in ("low", "medium", "high"):
        priority = "medium"

    escalation_id = _new_id("ESC")
    with get_db() as conn:
        conn.execute(
            """INSERT INTO escalations (escalation_id, customer_id, order_id, reason, priority, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?)""",
            (escalation_id, customer_id, order_id, reason, priority, datetime.now().isoformat()),
        )

    return {
        "escalation_id": escalation_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "reason": reason,
        "priority": priority,
        "status": "pending",
        "message": (
            f"Escalated to human support (ID {escalation_id}, priority {priority}). "
            f"A representative will contact you within 24 hours."
        ),
    }


def detect_sentiment(recent_messages: list[str]) -> dict:
    """Analyze recent customer messages for frustration. Uses Claude Haiku for speed and cost."""
    if not recent_messages:
        return {"sentiment": "neutral", "confidence": 0.0, "escalate": False}

    joined = "\n".join(f"- {m}" for m in recent_messages[-3:])  # last 3

    response = _anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                "Analyze the customer's sentiment and frustration level based on these recent messages.\n\n"
                f"Messages:\n{joined}\n\n"
                "Respond with ONLY a JSON object, no other text:\n"
                '{"sentiment": "positive"|"neutral"|"frustrated"|"angry", '
                '"confidence": 0.0-1.0, '
                '"escalate": true|false, '
                '"reasoning": "brief"}\n\n'
                "Set escalate=true if sentiment is frustrated or angry with confidence >= 0.7."
            ),
        }],
    )

    text = response.content[0].text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[1].rsplit("\n", 1)[0]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "escalate": False,
            "reasoning": "Sentiment analysis failed to parse.",
        }


# ==================== Tool schemas (what Claude sees) ====================

TOOLS = [
    {
        "name": "lookup_order",
        "description": (
            "Look up an order by its order ID. Returns order details including status, product, "
            "prices, delivery dates, and return validity date. Use whenever the customer mentions an order ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order ID in format 'ORD-XXXXXX'"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "lookup_customer_orders",
        "description": (
            "Look up all orders for a customer. Use when customer asks about their order history, "
            "or when they don't remember an order ID and provide a customer ID instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "description": "Customer ID in format 'CUST-XXXX'"},
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "check_shipping",
        "description": (
            "Get current shipping status and tracking information for an order. "
            "Use when the customer asks 'where is my order' or 'when will it arrive'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_return_policy",
        "description": (
            "Search the return policy document for relevant information. "
            "ALWAYS call this before deciding whether an order qualifies for a refund, "
            "or when the customer asks a policy question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language question about the policy"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "process_refund",
        "description": (
            "Process a refund automatically. "
            "IMPORTANT: Do NOT call this tool if ANY of the following are true:\n"
            "  - Refund amount is $50 or more\n"
            "  - Product is perishable (is_perishable=true in the order)\n"
            "  - Order was delivered more than 30 days ago\n"
            "In those cases, call escalate_to_human instead. "
            "Only use this tool for straightforward refunds within all three thresholds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number", "description": "Refund amount in USD (must be positive)"},
                "reason": {"type": "string", "description": "Reason for the refund"},
            },
            "required": ["order_id", "amount", "reason"],
        },
    },
    {
        "name": "cancel_order",
        "description": (
            "Cancel an order that has NOT yet shipped. "
            "Only works for orders in 'placed' or 'in-progress' status. "
            "Will fail (return an error) for orders that are already 'in-flight', 'delivered', or 'lost' — "
            "in those cases, use process_refund or escalate_to_human instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "reason"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Escalate the conversation to a human support representative. "
            "USE THIS WHENEVER any of the following are true:\n"
            "  - Refund amount is $50 or more\n"
            "  - Product is perishable\n"
            "  - Order was delivered more than 30 days ago and customer wants a refund\n"
            "  - Customer explicitly asks to speak to a human/person/representative\n"
            "  - Customer sounds frustrated or angry (confirmed via detect_sentiment)\n"
            "  - Situation requires human judgment beyond your tools"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "reason": {"type": "string", "description": "Why escalation is needed"},
                "order_id": {"type": "string", "description": "Related order ID if applicable"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Priority: 'high' for angry customers or urgent issues, 'medium' otherwise",
                },
            },
            "required": ["customer_id", "reason"],
        },
    },
    {
        "name": "create_ticket",
        "description": (
            "Create a support ticket to log the conversation. "
            "Call this ONCE at the end of every conversation to record what happened. "
            "Category should be one of: refund, shipping, cancellation, complaint, inquiry, escalation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "subject": {"type": "string", "description": "One-line summary"},
                "category": {
                    "type": "string",
                    "enum": ["refund", "shipping", "cancellation", "complaint", "inquiry", "escalation"],
                },
                "details": {"type": "string", "description": "Full details of the conversation and outcome"},
                "order_id": {"type": "string", "description": "Related order ID if applicable"},
            },
            "required": ["customer_id", "subject", "category", "details"],
        },
    },
    {
        "name": "detect_sentiment",
        "description": (
            "Analyze recent customer messages for frustration or anger. "
            "Call this when a customer's tone seems negative or when they've had to repeat themselves. "
            "If the returned 'escalate' field is true, follow up with escalate_to_human."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recent_messages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The 2-3 most recent customer messages",
                },
            },
            "required": ["recent_messages"],
        },
    },
]


# ==================== Dispatcher (routes tool_use to functions) ====================

_TOOL_REGISTRY = {
    "lookup_order":            lookup_order,
    "lookup_customer_orders":  lookup_customer_orders,
    "check_shipping":          check_shipping,
    "get_return_policy":       get_return_policy,
    "process_refund":          process_refund,
    "cancel_order":            cancel_order,
    "escalate_to_human":       escalate_to_human,
    "create_ticket":           create_ticket,
    "detect_sentiment":        detect_sentiment,
}


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route a tool_use block from Claude to the matching Python function."""
    fn = _TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: '{tool_name}'"}
    try:
        return fn(**tool_input)
    except TypeError as e:
        return {"error": f"Invalid arguments for '{tool_name}': {e}"}
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {type(e).__name__}: {e}"}


# ==================== Quick test ====================

if __name__ == '__main__':
    import pprint
    orders = _load_orders()

    print("=" * 60)
    print("Quick tool tests")
    print("=" * 60)

    # Grab representative order IDs for testing
    delivered_order = next(o for o in orders if o['order_status'] == 'delivered')
    in_flight_order = next(o for o in orders if o['order_status'] == 'in-flight')
    placed_order = next(o for o in orders if o['order_status'] == 'placed')
    perishable = next(
        (o for o in orders
         if o['is_perishable'] and o['order_status'] == 'delivered'),
        None,
    )

    tests = [
        ("lookup_order (real)", "lookup_order", {"order_id": delivered_order['order_id']}),
        ("lookup_order (fake)", "lookup_order", {"order_id": "ORD-000000"}),
        ("check_shipping (in-flight)", "check_shipping", {"order_id": in_flight_order['order_id']}),
        ("get_return_policy", "get_return_policy", {"query": "Can I return a perishable item?"}),
        ("cancel_order (allowed)", "cancel_order",
         {"order_id": placed_order['order_id'], "reason": "Changed my mind"}),
        ("cancel_order (blocked)", "cancel_order",
         {"order_id": delivered_order['order_id'], "reason": "Too late test"}),
        ("process_refund (small)", "process_refund",
         {"order_id": delivered_order['order_id'], "amount": 20.0, "reason": "Wrong size"}),
        ("escalate_to_human", "escalate_to_human",
         {"customer_id": delivered_order['customer_id'],
          "order_id": delivered_order['order_id'],
          "reason": "Refund exceeds $50",
          "priority": "medium"}),
        ("create_ticket", "create_ticket",
         {"customer_id": delivered_order['customer_id'],
          "subject": "Test ticket",
          "category": "inquiry",
          "details": "Testing the ticket creation flow"}),
        ("detect_sentiment (positive)", "detect_sentiment",
         {"recent_messages": ["Thanks so much for helping!", "You're great"]}),
        ("detect_sentiment (frustrated)", "detect_sentiment",
         {"recent_messages": ["This is ridiculous", "I've been waiting for 20 minutes", "Just refund my money already"]}),
    ]

    for label, name, args in tests:
        print(f"\n▶ {label}")
        print(f"  {name}({args})")
        result = dispatch_tool(name, args)
        pprint.pprint(result, indent=4, width=100, depth=3)
"""
Generates mock orders JSON for the customer support agent.
Run: python scripts/generate_mock_data.py
Output: data/orders.json
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

random.seed(42)  # reproducible runs

# Product catalog by category
CATEGORIES = {
    'electronics': {
        'is_perishable': False,
        'return_days': 30,
        'products': [
            ('Sony WH-1000XM5 Headphones', 349.99),
            ('Apple AirPods Pro', 249.00),
            ('Kindle Paperwhite', 149.99),
            ('Anker USB-C Hub', 39.99),
            ('Logitech MX Master 3S', 99.99),
        ],
    },
    'clothing': {
        'is_perishable': False,
        'return_days': 30,
        'products': [
            ('Nike Air Max 90', 130.00),
            ("Levi's 501 Jeans", 69.50),
            ('Uniqlo Merino Sweater', 49.90),
            ('Patagonia Fleece', 179.00),
            ('Champion Hoodie', 45.00),
        ],
    },
    'food_perishable': {
        'is_perishable': True,
        'return_days': 0,  # perishables typically non-returnable
        'products': [
            ('Fresh Blueberries 6oz', 5.99),
            ('Organic Milk 1 Gallon', 6.49),
            ('Sourdough Bread Loaf', 8.50),
            ('Rotisserie Chicken', 12.99),
            ('Fresh Salmon Fillet 1lb', 18.99),
        ],
    },
    'food_shelf_stable': {
        'is_perishable': False,
        'return_days': 30,
        'products': [
            ('Peanut Butter 32oz', 8.99),
            ('Organic Coffee Beans 1lb', 15.99),
            ('Pasta Sauce Jar 24oz', 4.99),
            ('Olive Oil 500ml', 12.99),
            ('Granola Cereal 16oz', 6.49),
        ],
    },
    'books': {
        'is_perishable': False,
        'return_days': 30,
        'products': [
            ('Designing Data-Intensive Applications', 55.00),
            ('The Pragmatic Programmer', 42.00),
            ('Clean Code', 35.00),
            ('System Design Interview Vol 1', 28.99),
            ('Atomic Habits', 18.99),
        ],
    },
    'beauty': {
        'is_perishable': False,
        'return_days': 15,  # opened cosmetics can't be returned; unopened has short window
        'products': [
            ('CeraVe Moisturizer', 18.99),
            ('The Ordinary Serum', 8.90),
            ('Sunscreen SPF 50', 24.99),
            ('Retinol Cream', 45.00),
            ('Hair Mask', 22.99),
        ],
    },
}

STATUSES = ['placed', 'in-progress', 'in-flight', 'delivered', 'lost']

# A small pool of customer IDs so some customers have multiple orders
CUSTOMER_POOL = [f"CUST-{i:04d}" for i in range(1001, 1021)]


def gen_id(prefix, digits=6):
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    return f"{prefix}-{random.randint(lo, hi)}"


def build_order(
    category_name=None,
    status=None,
    force_days_ago=None,
    force_price_bucket=None,   # 'under_50', 'over_50', or None
):
    """Build one order. Args override random choices for hand-crafted cases."""
    if category_name is None:
        category_name = random.choice(list(CATEGORIES.keys()))
    cat = CATEGORIES[category_name]

    # Pick product; optionally constrain by price
    products = cat['products']
    if force_price_bucket == 'under_50':
        products = [p for p in products if p[1] < 50]
    elif force_price_bucket == 'over_50':
        products = [p for p in products if p[1] >= 50]
    product_name, base_price = random.choice(products or cat['products'])

    quantity = random.randint(1, 3)
    discount_pct = random.choices([0, 5, 10, 15, 20], weights=[70, 10, 10, 5, 5])[0]
    product_cost = base_price * quantity
    total_cost = round(product_cost * (1 - discount_pct / 100), 2)

    if status is None:
        status = random.choices(
            STATUSES,
            weights=[5, 10, 15, 65, 5],  # most orders are delivered
        )[0]

    # Order received datetime
    days_ago = force_days_ago if force_days_ago is not None else random.randint(1, 60)
    order_received = datetime.now() - timedelta(
        days=days_ago,
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )

    # Estimated delivery: 3-7 days after order
    est_delivery = order_received + timedelta(days=random.randint(3, 7))

    # Delivered date (only if status = 'delivered')
    delivered_date = None
    if status == 'delivered':
        delivered_date = est_delivery + timedelta(days=random.randint(-1, 2))
        if delivered_date > datetime.now():
            delivered_date = datetime.now() - timedelta(days=random.randint(1, 5))

    # Return validity window
    return_validity = None
    if delivered_date and cat['return_days'] > 0:
        return_validity = delivered_date + timedelta(days=cat['return_days'])

    return {
        'order_id': gen_id('ORD'),
        'customer_id': random.choice(CUSTOMER_POOL),
        'product_id': gen_id('PRD', 5),
        'product_code': f"{category_name[:3].upper()}-{random.randint(1000, 9999)}",
        'product_name': product_name,
        'product_category': category_name,
        'is_perishable': cat['is_perishable'],
        'order_received_datetime': order_received.isoformat(timespec='seconds'),
        'estimated_delivery_date': est_delivery.date().isoformat(),
        'order_delivered_date': (
            delivered_date.date().isoformat() if delivered_date else None
        ),
        'order_status': status,
        'return_validity_date': (
            return_validity.date().isoformat() if return_validity else None
        ),
        'product_quantity': quantity,
        'product_cost': round(product_cost, 2),
        'discount_percentage': discount_pct,
        'total_cost': total_cost,
    }


def build_guaranteed_edge_cases():
    """Hand-crafted orders that guarantee each escalation trigger fires."""
    return [
        # 1. Delivered, under $50, within 30 days — HAPPY PATH refund
        build_order(
            category_name='books',
            status='delivered',
            force_days_ago=10,
            force_price_bucket='under_50',
        ),
        # 2. Delivered, over $50, within 30 days — REFUND-VALUE ESCALATION (>$50)
        build_order(
            category_name='electronics',
            status='delivered',
            force_days_ago=8,
            force_price_bucket='over_50',
        ),
        # 3. Delivered, perishable — PERISHABLE ESCALATION
        build_order(
            category_name='food_perishable',
            status='delivered',
            force_days_ago=3,
        ),
        # 4. Delivered, past 30 days — PAST-WINDOW ESCALATION
        build_order(
            category_name='clothing',
            status='delivered',
            force_days_ago=45,
            force_price_bucket='under_50',
        ),
        # 5. In-flight (in transit) — SHIPPING STATUS PATH
        build_order(
            category_name='electronics',
            status='in-flight',
            force_days_ago=4,
        ),
        # 6. Placed but not shipped — CANCELLATION PATH
        build_order(
            category_name='clothing',
            status='placed',
            force_days_ago=1,
        ),
        # 7. Lost — SPECIAL HANDLING PATH
        build_order(
            category_name='electronics',
            status='lost',
            force_days_ago=15,
            force_price_bucket='over_50',
        ),
        # 8. In-progress — STATUS QUERY PATH
        build_order(
            category_name='beauty',
            status='in-progress',
            force_days_ago=2,
        ),
    ]


def main():
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'orders.json'

    # 42 random orders + 8 hand-crafted edge cases = 50 total
    orders = [build_order() for _ in range(42)]
    orders.extend(build_guaranteed_edge_cases())

    # Save
    with open(output_file, 'w') as f:
        json.dump(orders, f, indent=2)

    # Report coverage
    print(f"Generated {len(orders)} orders → {output_file}\n")

    print("Status distribution:")
    for status, count in Counter(o['order_status'] for o in orders).most_common():
        print(f"  {status}: {count}")

    print("\nCategory distribution:")
    for cat, count in Counter(o['product_category'] for o in orders).most_common():
        print(f"  {cat}: {count}")

    delivered = [o for o in orders if o['order_status'] == 'delivered']
    print(f"\nRefund-eligibility spread across {len(delivered)} delivered orders:")
    print(f"  Under $50: {sum(1 for o in delivered if o['total_cost'] < 50)}")
    print(f"  Over $50:  {sum(1 for o in delivered if o['total_cost'] >= 50)}")
    print(f"  Perishable: {sum(1 for o in delivered if o['is_perishable'])}")


if __name__ == '__main__':
    main()
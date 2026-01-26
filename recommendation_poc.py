"""
Neo4j Recommendation Engine PoC

This script demonstrates a simple recommendation engine using Neo4j graph database.
Data model:
- User nodes: people who buy/rate products
- Product nodes: items that can be purchased/rated
- Category nodes: product categories
- PURCHASED relationship: User -> Product (with timestamp)
- RATED relationship: User -> Product (with rating 1-5)
- BELONGS_TO relationship: Product -> Category
"""

import random
from datetime import datetime, timedelta

from neo4j import GraphDatabase


class RecommendationEngine:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Remove all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def create_sample_data(self, num_users=100):
        """Create sample users, products, categories, and relationships at scale."""
        with self.driver.session() as session:
            # Create categories
            categories = [
                ('Electronics', ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch',
                                 'Camera', 'Monitor', 'Keyboard', 'Mouse', 'Speaker']),
                ('Books', ['Python Programming', 'Graph Databases', 'Machine Learning',
                          'Data Science', 'Web Development', 'Cloud Computing',
                          'Algorithms', 'System Design', 'DevOps', 'Cybersecurity']),
                ('Clothing', ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hoodie',
                             'Shorts', 'Sweater', 'Cap', 'Socks', 'Backpack']),
                ('Home & Kitchen', ['Coffee Maker', 'Blender', 'Air Fryer', 'Vacuum',
                                    'Lamp', 'Pillow', 'Blanket', 'Mug Set', 'Knife Set', 'Pan']),
                ('Sports', ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Water Bottle',
                           'Fitness Tracker', 'Resistance Bands', 'Jump Rope',
                           'Foam Roller', 'Gym Bag', 'Protein Shaker'])
            ]

            # Price ranges by category
            price_ranges = {
                'Electronics': (99.99, 1299.99),
                'Books': (19.99, 59.99),
                'Clothing': (24.99, 149.99),
                'Home & Kitchen': (29.99, 299.99),
                'Sports': (14.99, 129.99)
            }

            # Create categories and products
            print("Creating categories and products...")
            product_id = 1
            for cat_name, products in categories:
                session.run("CREATE (c:Category {name: $name})", name=cat_name)

                min_price, max_price = price_ranges[cat_name]
                for prod_name in products:
                    price = round(random.uniform(min_price, max_price), 2)
                    session.run("""
                        MATCH (c:Category {name: $cat_name})
                        CREATE (p:Product {id: $id, name: $name, price: $price})
                        CREATE (p)-[:BELONGS_TO]->(c)
                    """, id=f'P{product_id:03d}', name=prod_name, price=price, cat_name=cat_name)
                    product_id += 1

            total_products = product_id - 1
            print(f"  Created 5 categories and {total_products} products")

            # First names for generating user names
            first_names = [
                'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry',
                'Ivy', 'Jack', 'Kate', 'Liam', 'Mia', 'Noah', 'Olivia', 'Peter',
                'Quinn', 'Rose', 'Sam', 'Tina', 'Uma', 'Victor', 'Wendy', 'Xavier',
                'Yuki', 'Zoe', 'Adam', 'Beth', 'Carl', 'Donna', 'Eric', 'Fiona',
                'George', 'Hannah', 'Ian', 'Julia', 'Kevin', 'Laura', 'Mike', 'Nancy',
                'Oscar', 'Paula', 'Ray', 'Sara', 'Tom', 'Ursula', 'Vince', 'Wanda',
                'Xena', 'Yolanda', 'Zack', 'Amy', 'Brian', 'Chloe', 'Dan', 'Emma'
            ]

            # User personas (category preferences with weights)
            personas = [
                {'name': 'tech_enthusiast', 'weights': {'Electronics': 0.6, 'Books': 0.25, 'Sports': 0.1, 'Clothing': 0.05}},
                {'name': 'bookworm', 'weights': {'Books': 0.7, 'Home & Kitchen': 0.2, 'Clothing': 0.1}},
                {'name': 'fitness_fan', 'weights': {'Sports': 0.5, 'Clothing': 0.25, 'Electronics': 0.15, 'Home & Kitchen': 0.1}},
                {'name': 'homebody', 'weights': {'Home & Kitchen': 0.5, 'Books': 0.2, 'Clothing': 0.2, 'Electronics': 0.1}},
                {'name': 'fashionista', 'weights': {'Clothing': 0.6, 'Sports': 0.2, 'Electronics': 0.1, 'Home & Kitchen': 0.1}},
                {'name': 'generalist', 'weights': {'Electronics': 0.2, 'Books': 0.2, 'Clothing': 0.2, 'Home & Kitchen': 0.2, 'Sports': 0.2}},
            ]

            # Create users
            print(f"Creating {num_users} users...")
            for i in range(1, num_users + 1):
                name = f"{random.choice(first_names)}_{i}"
                session.run(
                    "CREATE (u:User {id: $id, name: $name})",
                    id=f'U{i:03d}', name=name
                )
            print(f"  Created {num_users} users")

            # Generate purchases and ratings
            print("Generating purchases and ratings...")
            base_date = datetime(2024, 1, 1)
            total_purchases = 0
            total_ratings = 0

            # Get all products grouped by category
            products_by_category = {}
            for cat_name, products in categories:
                result = session.run("""
                    MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $cat_name})
                    RETURN p.id AS id
                """, cat_name=cat_name)
                products_by_category[cat_name] = [r['id'] for r in result]

            for i in range(1, num_users + 1):
                user_id = f'U{i:03d}'
                persona = random.choice(personas)

                # Each user makes 3-12 purchases
                num_purchases = random.randint(3, 12)
                purchased_products = set()

                for _ in range(num_purchases):
                    # Select category based on persona weights
                    cat_name = random.choices(
                        list(persona['weights'].keys()),
                        weights=list(persona['weights'].values())
                    )[0]

                    # Select random product from category
                    available = [p for p in products_by_category[cat_name] if p not in purchased_products]
                    if not available:
                        continue

                    product_id = random.choice(available)
                    purchased_products.add(product_id)

                    # Random purchase date in 2024
                    purchase_date = base_date + timedelta(days=random.randint(0, 365))

                    session.run("""
                        MATCH (u:User {id: $user_id})
                        MATCH (p:Product {id: $product_id})
                        CREATE (u)-[:PURCHASED {date: $date}]->(p)
                    """, user_id=user_id, product_id=product_id, date=purchase_date.strftime('%Y-%m-%d'))
                    total_purchases += 1

                    # 60% chance to rate a purchased product
                    if random.random() < 0.6:
                        # Rating tends to be positive (3-5) with occasional low ratings
                        rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.35, 0.3])[0]
                        session.run("""
                            MATCH (u:User {id: $user_id})
                            MATCH (p:Product {id: $product_id})
                            CREATE (u)-[:RATED {rating: $rating}]->(p)
                        """, user_id=user_id, product_id=product_id, rating=rating)
                        total_ratings += 1

            print(f"  Created {total_purchases} purchases and {total_ratings} ratings")
            print("Sample data created successfully.")

    def get_collaborative_recommendations(self, user_name, limit=5):
        """
        Collaborative filtering: Find products purchased by similar users.
        Similar users = users who purchased the same products.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (user:User {name: $user_name})-[:PURCHASED]->(product:Product)
                      <-[:PURCHASED]-(similar_user:User)-[:PURCHASED]->(rec:Product)
                WHERE NOT (user)-[:PURCHASED]->(rec)
                  AND user <> similar_user
                WITH rec, COUNT(DISTINCT similar_user) AS score,
                     COLLECT(DISTINCT similar_user.name) AS recommenders
                RETURN rec.name AS product, rec.price AS price,
                       score, recommenders
                ORDER BY score DESC
                LIMIT $limit
            """, user_name=user_name, limit=limit)

            recommendations = list(result)
            print(f"\nCollaborative recommendations for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']}) - "
                      f"Score: {rec['score']}, Recommended by: {rec['recommenders']}")
            return recommendations

    def get_category_recommendations(self, user_name, limit=5):
        """
        Content-based filtering: Recommend products from categories
        the user has shown interest in.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (user:User {name: $user_name})-[:PURCHASED]->(:Product)
                      -[:BELONGS_TO]->(cat:Category)<-[:BELONGS_TO]-(rec:Product)
                WHERE NOT (user)-[:PURCHASED]->(rec)
                WITH rec, cat, COUNT(*) AS category_affinity
                RETURN rec.name AS product, rec.price AS price,
                       cat.name AS category, category_affinity
                ORDER BY category_affinity DESC
                LIMIT $limit
            """, user_name=user_name, limit=limit)

            recommendations = list(result)
            print(f"\nCategory-based recommendations for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']}) - "
                      f"Category: {rec['category']}, Affinity: {rec['category_affinity']}")
            return recommendations

    def get_highly_rated_in_category(self, category_name, min_rating=4, limit=5):
        """Find highly-rated products in a specific category."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category})
                MATCH (u:User)-[r:RATED]->(p)
                WHERE r.rating >= $min_rating
                WITH p, AVG(r.rating) AS avg_rating, COUNT(r) AS num_ratings
                RETURN p.name AS product, p.price AS price,
                       round(avg_rating * 100) / 100 AS avg_rating, num_ratings
                ORDER BY avg_rating DESC, num_ratings DESC
                LIMIT $limit
            """, category=category_name, min_rating=min_rating, limit=limit)

            products = list(result)
            print(f"\nTop rated products in {category_name}:")
            print("-" * 60)
            if not products:
                print("No products found.")
            for p in products:
                print(f"  {p['product']} (${p['price']}) - "
                      f"Avg rating: {p['avg_rating']}, Reviews: {p['num_ratings']}")
            return products

    def get_purchase_path(self, user1_name, user2_name):
        """Find common products between two users (shared interests)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u1:User {name: $user1})-[:PURCHASED]->(p:Product)
                      <-[:PURCHASED]-(u2:User {name: $user2})
                RETURN p.name AS product, p.price AS price
            """, user1=user1_name, user2=user2_name)

            common = list(result)
            print(f"\nCommon purchases between {user1_name} and {user2_name}:")
            print("-" * 60)
            if not common:
                print("No common purchases.")
            for p in common:
                print(f"  {p['product']} (${p['price']})")
            return common

    def get_trending_products(self, days=90, limit=10):
        """
        Find trending products based on recent purchase activity.
        Products with more purchases in the recent time window rank higher.
        """
        with self.driver.session() as session:
            # Calculate the cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            result = session.run("""
                MATCH (u:User)-[p:PURCHASED]->(prod:Product)-[:BELONGS_TO]->(c:Category)
                WHERE p.date >= $cutoff_date
                WITH prod, c, COUNT(p) AS recent_purchases,
                     COUNT(DISTINCT u) AS unique_buyers
                OPTIONAL MATCH (u2:User)-[r:RATED]->(prod)
                WITH prod, c, recent_purchases, unique_buyers,
                     AVG(r.rating) AS avg_rating
                RETURN prod.name AS product, prod.price AS price,
                       c.name AS category, recent_purchases, unique_buyers,
                       COALESCE(round(avg_rating * 100) / 100, 0) AS avg_rating
                ORDER BY recent_purchases DESC, unique_buyers DESC
                LIMIT $limit
            """, cutoff_date=cutoff_date, limit=limit)

            products = list(result)
            print(f"\nTrending Products (last {days} days):")
            print("-" * 60)
            if not products:
                print("No trending products found.")
            for p in products:
                rating_str = f", Rating: {p['avg_rating']}" if p['avg_rating'] > 0 else ""
                print(f"  {p['product']} (${p['price']}) - {p['category']}")
                print(f"      Purchases: {p['recent_purchases']}, Unique buyers: {p['unique_buyers']}{rating_str}")
            return products

    def get_time_decay_recommendations(self, user_name, decay_days=180, limit=5):
        """
        Recommend products with time-decay weighting.
        More recent purchases by similar users are weighted higher.
        """
        with self.driver.session() as session:
            today = datetime.now().strftime('%Y-%m-%d')

            result = session.run("""
                MATCH (user:User {name: $user_name})-[:PURCHASED]->(product:Product)
                      <-[p1:PURCHASED]-(similar_user:User)-[p2:PURCHASED]->(rec:Product)
                WHERE NOT (user)-[:PURCHASED]->(rec)
                  AND user <> similar_user
                WITH rec, similar_user, p2.date AS purchase_date,
                     duration.inDays(date(p2.date), date($today)).days AS days_ago
                WITH rec, similar_user,
                     CASE WHEN days_ago < 0 THEN 0 ELSE days_ago END AS days_ago
                WITH rec, similar_user,
                     // Exponential decay: more recent = higher weight
                     exp(-toFloat(days_ago) / $decay_days) AS time_weight
                WITH rec, SUM(time_weight) AS weighted_score,
                     COUNT(DISTINCT similar_user) AS recommender_count,
                     COLLECT(DISTINCT similar_user.name)[0..5] AS sample_recommenders
                RETURN rec.name AS product, rec.price AS price,
                       round(weighted_score * 100) / 100 AS weighted_score,
                       recommender_count, sample_recommenders
                ORDER BY weighted_score DESC
                LIMIT $limit
            """, user_name=user_name, today=today, decay_days=float(decay_days), limit=limit)

            recommendations = list(result)
            print(f"\nTime-weighted recommendations for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']})")
                print(f"      Score: {rec['weighted_score']}, From {rec['recommender_count']} users: {rec['sample_recommenders']}")
            return recommendations

    def get_user_similarity(self, user_name, limit=10):
        """
        Calculate Jaccard similarity between users based on purchase patterns.
        Jaccard = |intersection| / |union| of purchased products.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (user:User {name: $user_name})-[:PURCHASED]->(p:Product)
                WITH user, COLLECT(p) AS user_products, COUNT(p) AS user_count

                MATCH (other:User)-[:PURCHASED]->(p2:Product)
                WHERE other <> user
                WITH user, user_products, user_count, other, COLLECT(p2) AS other_products, COUNT(p2) AS other_count

                // Calculate intersection (common products)
                WITH user, other, user_products, other_products, user_count, other_count,
                     [p IN user_products WHERE p IN other_products] AS intersection

                // Jaccard similarity = intersection / union
                WITH user, other,
                     SIZE(intersection) AS common_count,
                     user_count + other_count - SIZE(intersection) AS union_count,
                     SIZE(intersection) AS intersection_size

                WHERE intersection_size > 0
                WITH other,
                     toFloat(common_count) / union_count AS jaccard_similarity,
                     common_count

                RETURN other.name AS similar_user,
                       round(jaccard_similarity * 1000) / 1000 AS similarity,
                       common_count AS common_products
                ORDER BY jaccard_similarity DESC
                LIMIT $limit
            """, user_name=user_name, limit=limit)

            similar_users = list(result)
            print(f"\nMost similar users to {user_name} (Jaccard similarity):")
            print("-" * 60)
            if not similar_users:
                print("No similar users found.")
            for u in similar_users:
                print(f"  {u['similar_user']} - Similarity: {u['similarity']:.3f} ({u['common_products']} common products)")
            return similar_users

    def get_similar_user_recommendations(self, user_name, top_n_users=5, limit=5):
        """
        Get recommendations from the most similar users.
        First find similar users, then recommend their highly-rated products.
        """
        with self.driver.session() as session:
            result = session.run("""
                // Find most similar users using Jaccard
                MATCH (user:User {name: $user_name})-[:PURCHASED]->(p:Product)
                WITH user, COLLECT(p) AS user_products

                MATCH (other:User)-[:PURCHASED]->(p2:Product)
                WHERE other <> user
                WITH user, user_products, other, COLLECT(p2) AS other_products

                WITH user, other, user_products, other_products,
                     [p IN user_products WHERE p IN other_products] AS intersection
                WITH user, other,
                     toFloat(SIZE(intersection)) / (SIZE(user_products) + SIZE(other_products) - SIZE(intersection)) AS similarity
                WHERE similarity > 0
                ORDER BY similarity DESC
                LIMIT $top_n_users

                // Get products that similar users rated highly but target user hasn't purchased
                WITH user, COLLECT(other) AS similar_users, COLLECT(similarity) AS similarities

                UNWIND range(0, size(similar_users)-1) AS idx
                WITH user, similar_users[idx] AS similar_user, similarities[idx] AS sim

                MATCH (similar_user)-[r:RATED]->(rec:Product)
                WHERE r.rating >= 4
                  AND NOT (user)-[:PURCHASED]->(rec)
                WITH rec, SUM(sim * r.rating) AS weighted_rating,
                     COUNT(DISTINCT similar_user) AS recommender_count,
                     COLLECT(DISTINCT similar_user.name) AS recommenders

                RETURN rec.name AS product, rec.price AS price,
                       round(weighted_rating * 100) / 100 AS score,
                       recommender_count, recommenders
                ORDER BY weighted_rating DESC
                LIMIT $limit
            """, user_name=user_name, top_n_users=top_n_users, limit=limit)

            recommendations = list(result)
            print(f"\nRecommendations from similar users for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']})")
                print(f"      Score: {rec['score']}, From: {rec['recommenders']}")
            return recommendations

    def get_purchase_velocity(self, limit=10):
        """
        Find products with increasing purchase velocity (momentum).
        Compare recent purchases vs older purchases.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[p:PURCHASED]->(prod:Product)-[:BELONGS_TO]->(c:Category)
                WITH prod, c, p.date AS purchase_date,
                     CASE
                         WHEN p.date >= '2024-07-01' THEN 'recent'
                         ELSE 'older'
                     END AS period
                WITH prod, c, period, COUNT(*) AS count
                WITH prod, c,
                     SUM(CASE WHEN period = 'recent' THEN count ELSE 0 END) AS recent_count,
                     SUM(CASE WHEN period = 'older' THEN count ELSE 0 END) AS older_count
                WHERE older_count > 0
                WITH prod, c, recent_count, older_count,
                     toFloat(recent_count) / older_count AS velocity_ratio
                WHERE recent_count >= 2
                RETURN prod.name AS product, prod.price AS price, c.name AS category,
                       recent_count AS recent_purchases,
                       older_count AS older_purchases,
                       round(velocity_ratio * 100) / 100 AS velocity
                ORDER BY velocity_ratio DESC
                LIMIT $limit
            """, limit=limit)

            products = list(result)
            print("\nProducts with Increasing Momentum (H2 vs H1 2024):")
            print("-" * 60)
            if not products:
                print("No products with momentum data found.")
            for p in products:
                print(f"  {p['product']} ({p['category']}) - ${p['price']}")
                print(f"      Recent: {p['recent_purchases']}, Older: {p['older_purchases']}, Velocity: {p['velocity']}x")
            return products

    def get_user_purchase_profile(self, user_name):
        """
        Get detailed purchase profile for a user including category breakdown,
        spending patterns, and rating behavior.
        """
        with self.driver.session() as session:
            # Basic stats
            basic_stats = session.run("""
                MATCH (u:User {name: $user_name})
                OPTIONAL MATCH (u)-[p:PURCHASED]->(prod:Product)
                OPTIONAL MATCH (u)-[r:RATED]->(rated_prod:Product)
                WITH u,
                     COUNT(DISTINCT prod) AS total_purchases,
                     COALESCE(SUM(prod.price), 0) AS total_spent,
                     COUNT(DISTINCT rated_prod) AS products_rated,
                     AVG(r.rating) AS avg_rating_given
                RETURN total_purchases,
                       round(total_spent * 100) / 100 AS total_spent,
                       products_rated,
                       COALESCE(round(avg_rating_given * 100) / 100, 0) AS avg_rating_given
            """, user_name=user_name).single()

            # Category breakdown
            category_breakdown = session.run("""
                MATCH (u:User {name: $user_name})-[:PURCHASED]->(p:Product)-[:BELONGS_TO]->(c:Category)
                WITH c.name AS category, COUNT(p) AS purchases, SUM(p.price) AS spent
                RETURN category, purchases, round(spent * 100) / 100 AS spent
                ORDER BY purchases DESC
            """, user_name=user_name)

            categories = list(category_breakdown)

            print(f"\nPurchase Profile for {user_name}:")
            print("-" * 60)
            print(f"  Total purchases: {basic_stats['total_purchases']}")
            print(f"  Total spent: ${basic_stats['total_spent']}")
            print(f"  Products rated: {basic_stats['products_rated']}")
            print(f"  Average rating given: {basic_stats['avg_rating_given']}")
            print("\n  Category breakdown:")
            for cat in categories:
                print(f"    {cat['category']}: {cat['purchases']} purchases (${cat['spent']})")

            return {'basic': basic_stats, 'categories': categories}

    def show_graph_stats(self):
        """Display basic statistics about the graph."""
        with self.driver.session() as session:
            stats = session.run("""
                MATCH (u:User) WITH COUNT(u) AS users
                MATCH (p:Product) WITH users, COUNT(p) AS products
                MATCH (c:Category) WITH users, products, COUNT(c) AS categories
                MATCH ()-[pur:PURCHASED]->() WITH users, products, categories, COUNT(pur) AS purchases
                MATCH ()-[rat:RATED]->()
                RETURN users, products, categories, purchases, COUNT(rat) AS ratings
            """).single()

            print("\nGraph Statistics:")
            print("-" * 60)
            print(f"  Users: {stats['users']}")
            print(f"  Products: {stats['products']}")
            print(f"  Categories: {stats['categories']}")
            print(f"  Purchases: {stats['purchases']}")
            print(f"  Ratings: {stats['ratings']}")


def main():
    # Connection settings for Neo4j Desktop
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "airspace"

    print("=" * 60)
    print("Neo4j Recommendation Engine PoC (100 Users Scale)")
    print("=" * 60)

    try:
        engine = RecommendationEngine(URI, USER, PASSWORD)

        # Setup: Clear and populate database with 100 users
        engine.clear_database()
        engine.create_sample_data(num_users=100)

        # Show graph statistics
        engine.show_graph_stats()

        # Get some random users to demo with
        with engine.driver.session() as session:
            result = session.run("""
                MATCH (u:User)
                RETURN u.name AS name
                ORDER BY rand()
                LIMIT 4
            """)
            sample_users = [r['name'] for r in result]

        print(f"\nDemo users: {sample_users}")

        # === BASIC RECOMMENDATIONS ===
        print("\n" + "=" * 60)
        print("BASIC RECOMMENDATIONS")
        print("=" * 60)

        # Demo 1: Collaborative filtering recommendations
        engine.get_collaborative_recommendations(sample_users[0])

        # Demo 2: Category-based recommendations
        engine.get_category_recommendations(sample_users[1])

        # Demo 3: Top rated products by category
        engine.get_highly_rated_in_category("Electronics")

        # === TIME-BASED FEATURES ===
        print("\n" + "=" * 60)
        print("TIME-BASED FEATURES")
        print("=" * 60)

        # Demo 4: Trending products
        engine.get_trending_products(days=180, limit=5)

        # Demo 5: Time-decay weighted recommendations
        engine.get_time_decay_recommendations(sample_users[0])

        # Demo 6: Purchase velocity (momentum)
        engine.get_purchase_velocity(limit=5)

        # === USER SIMILARITY FEATURES ===
        print("\n" + "=" * 60)
        print("USER SIMILARITY FEATURES")
        print("=" * 60)

        # Demo 7: Find similar users (Jaccard similarity)
        engine.get_user_similarity(sample_users[0], limit=5)

        # Demo 8: Recommendations from similar users
        engine.get_similar_user_recommendations(sample_users[0])

        # Demo 9: User purchase profile
        engine.get_user_purchase_profile(sample_users[0])

        # === USER CONNECTIONS ===
        print("\n" + "=" * 60)
        print("USER CONNECTIONS")
        print("=" * 60)

        # Demo 10: Common purchases between users
        if len(sample_users) >= 2:
            engine.get_purchase_path(sample_users[0], sample_users[1])

        engine.close()
        print("\n" + "=" * 60)
        print("PoC completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Neo4j Desktop is running")
        print("2. A database is started")
        print("3. Update the PASSWORD variable in main() to match your Neo4j password")


if __name__ == "__main__":
    main()

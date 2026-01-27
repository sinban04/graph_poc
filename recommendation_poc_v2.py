"""
Neo4j Recommendation Engine PoC v2 - Enhanced

Enhanced version with additional graph algorithms and features:
- Original recommendation algorithms
- PageRank-like influence scoring
- Community detection (label propagation concept)
- Product co-purchase analysis (market basket)
- User journey/path analysis
- Fraud detection patterns (unusual rating behavior)
- A/B testing simulation for recommendations

Data model:
- User nodes: people who buy/rate products
- Product nodes: items that can be purchased/rated
- Category nodes: product categories
- PURCHASED relationship: User -> Product (with timestamp)
- RATED relationship: User -> Product (with rating 1-5)
- VIEWED relationship: User -> Product (with timestamp, duration)
- WISHLISTED relationship: User -> Product (with timestamp)
- BELONGS_TO relationship: Product -> Category
- SIMILAR_TO relationship: Product -> Product (computed similarity)
"""

import random
from datetime import datetime, timedelta
from collections import defaultdict

from neo4j import GraphDatabase


class RecommendationEngineV2:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Remove all nodes and relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def create_enhanced_sample_data(self, num_users=100):
        """Create enhanced sample data with more relationship types."""
        with self.driver.session() as session:
            # Create categories with descriptions
            categories = [
                ('Electronics', 'Electronic devices and gadgets',
                 ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smartwatch',
                  'Camera', 'Monitor', 'Keyboard', 'Mouse', 'Speaker']),
                ('Books', 'Technical and educational books',
                 ['Python Programming', 'Graph Databases', 'Machine Learning',
                  'Data Science', 'Web Development', 'Cloud Computing',
                  'Algorithms', 'System Design', 'DevOps', 'Cybersecurity']),
                ('Clothing', 'Apparel and accessories',
                 ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Hoodie',
                  'Shorts', 'Sweater', 'Cap', 'Socks', 'Backpack']),
                ('Home & Kitchen', 'Home appliances and kitchenware',
                 ['Coffee Maker', 'Blender', 'Air Fryer', 'Vacuum',
                  'Lamp', 'Pillow', 'Blanket', 'Mug Set', 'Knife Set', 'Pan']),
                ('Sports', 'Fitness and outdoor equipment',
                 ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Water Bottle',
                  'Fitness Tracker', 'Resistance Bands', 'Jump Rope',
                  'Foam Roller', 'Gym Bag', 'Protein Shaker'])
            ]

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
            for cat_name, cat_desc, products in categories:
                session.run(
                    "CREATE (c:Category {name: $name, description: $desc})",
                    name=cat_name, desc=cat_desc
                )

                min_price, max_price = price_ranges[cat_name]
                for prod_name in products:
                    price = round(random.uniform(min_price, max_price), 2)
                    session.run("""
                        MATCH (c:Category {name: $cat_name})
                        CREATE (p:Product {
                            id: $id,
                            name: $name,
                            price: $price,
                            created_date: $created
                        })
                        CREATE (p)-[:BELONGS_TO]->(c)
                    """, id=f'P{product_id:03d}', name=prod_name, price=price,
                        cat_name=cat_name, created='2023-01-01')
                    product_id += 1

            total_products = product_id - 1
            print(f"  Created 5 categories and {total_products} products")

            # User personas with more detailed behavior patterns
            personas = [
                {'name': 'tech_enthusiast',
                 'weights': {'Electronics': 0.6, 'Books': 0.25, 'Sports': 0.1, 'Clothing': 0.05},
                 'view_rate': 0.8, 'wishlist_rate': 0.3, 'purchase_rate': 0.4},
                {'name': 'bookworm',
                 'weights': {'Books': 0.7, 'Home & Kitchen': 0.2, 'Clothing': 0.1},
                 'view_rate': 0.9, 'wishlist_rate': 0.5, 'purchase_rate': 0.6},
                {'name': 'fitness_fan',
                 'weights': {'Sports': 0.5, 'Clothing': 0.25, 'Electronics': 0.15, 'Home & Kitchen': 0.1},
                 'view_rate': 0.7, 'wishlist_rate': 0.2, 'purchase_rate': 0.5},
                {'name': 'homebody',
                 'weights': {'Home & Kitchen': 0.5, 'Books': 0.2, 'Clothing': 0.2, 'Electronics': 0.1},
                 'view_rate': 0.6, 'wishlist_rate': 0.4, 'purchase_rate': 0.5},
                {'name': 'fashionista',
                 'weights': {'Clothing': 0.6, 'Sports': 0.2, 'Electronics': 0.1, 'Home & Kitchen': 0.1},
                 'view_rate': 0.85, 'wishlist_rate': 0.6, 'purchase_rate': 0.3},
                {'name': 'generalist',
                 'weights': {'Electronics': 0.2, 'Books': 0.2, 'Clothing': 0.2, 'Home & Kitchen': 0.2, 'Sports': 0.2},
                 'view_rate': 0.5, 'wishlist_rate': 0.2, 'purchase_rate': 0.4},
            ]

            first_names = [
                'Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry',
                'Ivy', 'Jack', 'Kate', 'Liam', 'Mia', 'Noah', 'Olivia', 'Peter',
                'Quinn', 'Rose', 'Sam', 'Tina', 'Uma', 'Victor', 'Wendy', 'Xavier',
                'Yuki', 'Zoe', 'Adam', 'Beth', 'Carl', 'Donna', 'Eric', 'Fiona',
                'George', 'Hannah', 'Ian', 'Julia', 'Kevin', 'Laura', 'Mike', 'Nancy',
                'Oscar', 'Paula', 'Ray', 'Sara', 'Tom', 'Ursula', 'Vince', 'Wanda'
            ]

            # Create users with persona
            print(f"Creating {num_users} users...")
            user_personas = {}
            for i in range(1, num_users + 1):
                name = f"{random.choice(first_names)}_{i}"
                persona = random.choice(personas)
                user_personas[f'U{i:03d}'] = persona
                session.run(
                    "CREATE (u:User {id: $id, name: $name, persona: $persona})",
                    id=f'U{i:03d}', name=name, persona=persona['name']
                )
            print(f"  Created {num_users} users")

            # Get products by category
            products_by_category = {}
            for cat_name, _, products in categories:
                result = session.run("""
                    MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $cat_name})
                    RETURN p.id AS id
                """, cat_name=cat_name)
                products_by_category[cat_name] = [r['id'] for r in result]

            # Generate interactions
            print("Generating user interactions (views, wishlists, purchases, ratings)...")
            base_date = datetime(2024, 1, 1)
            stats = {'views': 0, 'wishlists': 0, 'purchases': 0, 'ratings': 0}

            for i in range(1, num_users + 1):
                user_id = f'U{i:03d}'
                persona = user_personas[user_id]

                # Generate views (users view many products)
                num_views = random.randint(10, 30)
                viewed_products = set()

                for _ in range(num_views):
                    cat_name = random.choices(
                        list(persona['weights'].keys()),
                        weights=list(persona['weights'].values())
                    )[0]

                    available = products_by_category.get(cat_name, [])
                    if not available:
                        continue

                    product_id = random.choice(available)
                    viewed_products.add(product_id)

                    view_date = base_date + timedelta(days=random.randint(0, 365))
                    duration = random.randint(5, 300)  # seconds

                    session.run("""
                        MATCH (u:User {id: $user_id})
                        MATCH (p:Product {id: $product_id})
                        MERGE (u)-[v:VIEWED]->(p)
                        ON CREATE SET v.first_view = $date, v.duration = $duration, v.view_count = 1
                        ON MATCH SET v.view_count = v.view_count + 1, v.duration = v.duration + $duration
                    """, user_id=user_id, product_id=product_id,
                        date=view_date.strftime('%Y-%m-%d'), duration=duration)
                    stats['views'] += 1

                # Wishlists (subset of viewed)
                wishlist_count = int(len(viewed_products) * persona['wishlist_rate'])
                wishlisted = random.sample(list(viewed_products), min(wishlist_count, len(viewed_products)))

                for product_id in wishlisted:
                    wishlist_date = base_date + timedelta(days=random.randint(0, 365))
                    session.run("""
                        MATCH (u:User {id: $user_id})
                        MATCH (p:Product {id: $product_id})
                        CREATE (u)-[:WISHLISTED {date: $date}]->(p)
                    """, user_id=user_id, product_id=product_id,
                        date=wishlist_date.strftime('%Y-%m-%d'))
                    stats['wishlists'] += 1

                # Purchases (subset of viewed, bias toward wishlisted)
                purchase_count = random.randint(3, 12)
                purchased = set()

                # Higher chance to purchase wishlisted items
                purchase_pool = list(wishlisted) + list(viewed_products - set(wishlisted))
                weights = [2.0] * len(wishlisted) + [1.0] * len(viewed_products - set(wishlisted))

                for _ in range(min(purchase_count, len(purchase_pool))):
                    if not purchase_pool:
                        break
                    idx = random.choices(range(len(purchase_pool)), weights=weights)[0]
                    product_id = purchase_pool.pop(idx)
                    weights.pop(idx)
                    purchased.add(product_id)

                    purchase_date = base_date + timedelta(days=random.randint(0, 365))
                    session.run("""
                        MATCH (u:User {id: $user_id})
                        MATCH (p:Product {id: $product_id})
                        CREATE (u)-[:PURCHASED {date: $date}]->(p)
                    """, user_id=user_id, product_id=product_id,
                        date=purchase_date.strftime('%Y-%m-%d'))
                    stats['purchases'] += 1

                    # Ratings for purchased items
                    if random.random() < 0.6:
                        rating = random.choices([1, 2, 3, 4, 5],
                                              weights=[0.05, 0.1, 0.2, 0.35, 0.3])[0]
                        session.run("""
                            MATCH (u:User {id: $user_id})
                            MATCH (p:Product {id: $product_id})
                            CREATE (u)-[:RATED {rating: $rating}]->(p)
                        """, user_id=user_id, product_id=product_id, rating=rating)
                        stats['ratings'] += 1

            print(f"  Created {stats['views']} views, {stats['wishlists']} wishlists, "
                  f"{stats['purchases']} purchases, {stats['ratings']} ratings")

            # Compute product similarity relationships
            print("Computing product similarities...")
            session.run("""
                MATCH (p1:Product)<-[:PURCHASED]-(u:User)-[:PURCHASED]->(p2:Product)
                WHERE p1.id < p2.id
                WITH p1, p2, COUNT(DISTINCT u) AS co_purchases
                WHERE co_purchases >= 3
                MERGE (p1)-[s:SIMILAR_TO]-(p2)
                SET s.strength = co_purchases
            """)

            sim_count = session.run("""
                MATCH ()-[s:SIMILAR_TO]-()
                RETURN COUNT(s) / 2 AS count
            """).single()['count']
            print(f"  Created {sim_count} product similarity relationships")

            print("Enhanced sample data created successfully.")

    def show_graph_stats(self):
        """Display statistics about the graph."""
        with self.driver.session() as session:
            stats = session.run("""
                MATCH (u:User) WITH COUNT(u) AS users
                MATCH (p:Product) WITH users, COUNT(p) AS products
                MATCH (c:Category) WITH users, products, COUNT(c) AS categories
                MATCH ()-[v:VIEWED]->() WITH users, products, categories, COUNT(v) AS views
                MATCH ()-[w:WISHLISTED]->() WITH users, products, categories, views, COUNT(w) AS wishlists
                MATCH ()-[pur:PURCHASED]->() WITH users, products, categories, views, wishlists, COUNT(pur) AS purchases
                MATCH ()-[rat:RATED]->() WITH users, products, categories, views, wishlists, purchases, COUNT(rat) AS ratings
                MATCH ()-[sim:SIMILAR_TO]-()
                RETURN users, products, categories, views, wishlists, purchases, ratings, COUNT(sim)/2 AS similarities
            """).single()

            print("\nGraph Statistics:")
            print("-" * 60)
            print(f"  Users: {stats['users']}")
            print(f"  Products: {stats['products']}")
            print(f"  Categories: {stats['categories']}")
            print(f"  Views: {stats['views']}")
            print(f"  Wishlists: {stats['wishlists']}")
            print(f"  Purchases: {stats['purchases']}")
            print(f"  Ratings: {stats['ratings']}")
            print(f"  Product Similarities: {stats['similarities']}")

    # ==================== ORIGINAL ALGORITHMS ====================

    def get_collaborative_recommendations(self, user_name, limit=5):
        """Collaborative filtering: Find products purchased by similar users."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (user:User {name: $user_name})-[:PURCHASED]->(product:Product)
                      <-[:PURCHASED]-(similar_user:User)-[:PURCHASED]->(rec:Product)
                WHERE NOT (user)-[:PURCHASED]->(rec)
                  AND user <> similar_user
                WITH rec, COUNT(DISTINCT similar_user) AS score,
                     COLLECT(DISTINCT similar_user.name)[0..5] AS recommenders
                RETURN rec.name AS product, rec.price AS price,
                       score, recommenders
                ORDER BY score DESC
                LIMIT $limit
            """, user_name=user_name, limit=limit)

            recommendations = list(result)
            print(f"\nCollaborative Filtering for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("  No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']}) - Score: {rec['score']}")
            return recommendations

    def get_category_recommendations(self, user_name, limit=5):
        """Content-based filtering using category preferences."""
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
            print(f"\nCategory-based Recommendations for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("  No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']}) - {rec['category']} (affinity: {rec['category_affinity']})")
            return recommendations

    # ==================== NEW: ENGAGEMENT-BASED ====================

    def get_view_to_purchase_funnel(self, user_name):
        """Analyze user's conversion funnel: views -> wishlists -> purchases."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {name: $user_name})
                OPTIONAL MATCH (u)-[v:VIEWED]->(viewed:Product)
                OPTIONAL MATCH (u)-[:WISHLISTED]->(wishlisted:Product)
                OPTIONAL MATCH (u)-[:PURCHASED]->(purchased:Product)
                WITH u,
                     COUNT(DISTINCT viewed) AS total_viewed,
                     COUNT(DISTINCT wishlisted) AS total_wishlisted,
                     COUNT(DISTINCT purchased) AS total_purchased
                RETURN total_viewed, total_wishlisted, total_purchased,
                       CASE WHEN total_viewed > 0
                            THEN round(toFloat(total_wishlisted) / total_viewed * 100)
                            ELSE 0 END AS wishlist_rate,
                       CASE WHEN total_viewed > 0
                            THEN round(toFloat(total_purchased) / total_viewed * 100)
                            ELSE 0 END AS conversion_rate
            """, user_name=user_name).single()

            print(f"\nConversion Funnel for {user_name}:")
            print("-" * 60)
            print(f"  Products Viewed: {result['total_viewed']}")
            print(f"  Products Wishlisted: {result['total_wishlisted']} ({result['wishlist_rate']}%)")
            print(f"  Products Purchased: {result['total_purchased']} ({result['conversion_rate']}%)")
            return result

    def get_wishlist_recommendations(self, user_name, limit=5):
        """Recommend products similar to user's wishlisted items."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (user:User {name: $user_name})-[:WISHLISTED]->(wish:Product)
                WHERE NOT (user)-[:PURCHASED]->(wish)

                // Find similar products to wishlisted items
                OPTIONAL MATCH (wish)-[s:SIMILAR_TO]-(similar:Product)
                WHERE NOT (user)-[:PURCHASED]->(similar)
                  AND NOT (user)-[:WISHLISTED]->(similar)

                WITH wish, similar, s.strength AS similarity
                WHERE similar IS NOT NULL

                RETURN similar.name AS product, similar.price AS price,
                       COLLECT(DISTINCT wish.name)[0..3] AS based_on,
                       SUM(similarity) AS total_similarity
                ORDER BY total_similarity DESC
                LIMIT $limit
            """, user_name=user_name, limit=limit)

            recommendations = list(result)
            print(f"\nWishlist-based Recommendations for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("  No recommendations found (try building more similarity data).")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']})")
                print(f"      Similar to: {rec['based_on']}")
            return recommendations

    # ==================== NEW: MARKET BASKET ANALYSIS ====================

    def get_frequently_bought_together(self, product_name, limit=5):
        """Find products frequently purchased together (market basket)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Product {name: $product_name})<-[:PURCHASED]-(u:User)-[:PURCHASED]->(other:Product)
                WHERE p <> other
                WITH other, COUNT(DISTINCT u) AS co_purchase_count

                // Get total purchasers of the target product
                MATCH (p:Product {name: $product_name})<-[:PURCHASED]-(buyer:User)
                WITH other, co_purchase_count, COUNT(DISTINCT buyer) AS total_buyers

                // Calculate lift (co-occurrence / expected)
                MATCH (all_buyer:User)-[:PURCHASED]->(other)
                WITH other, co_purchase_count, total_buyers,
                     COUNT(DISTINCT all_buyer) AS other_buyers

                RETURN other.name AS product, other.price AS price,
                       co_purchase_count,
                       round(toFloat(co_purchase_count) / total_buyers * 100) AS confidence,
                       other_buyers
                ORDER BY co_purchase_count DESC
                LIMIT $limit
            """, product_name=product_name, limit=limit)

            products = list(result)
            print(f"\nFrequently Bought Together with '{product_name}':")
            print("-" * 60)
            if not products:
                print("  No co-purchase data found.")
            for p in products:
                print(f"  {p['product']} (${p['price']}) - "
                      f"Co-purchased {p['co_purchase_count']} times ({p['confidence']}% confidence)")
            return products

    def get_bundle_suggestions(self, category_name, min_support=3, limit=5):
        """Suggest product bundles based on co-purchase patterns in a category."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p1:Product)-[:BELONGS_TO]->(c:Category {name: $category})
                MATCH (p1)<-[:PURCHASED]-(u:User)-[:PURCHASED]->(p2:Product)
                WHERE p1.id < p2.id
                WITH p1, p2, COUNT(DISTINCT u) AS support
                WHERE support >= $min_support

                // Calculate bundle value
                WITH p1, p2, support,
                     p1.price + p2.price AS bundle_price

                RETURN p1.name AS product1, p2.name AS product2,
                       support, round(bundle_price * 100) / 100 AS bundle_price
                ORDER BY support DESC
                LIMIT $limit
            """, category=category_name, min_support=min_support, limit=limit)

            bundles = list(result)
            print(f"\nSuggested Bundles in {category_name}:")
            print("-" * 60)
            if not bundles:
                print("  No bundle patterns found.")
            for b in bundles:
                print(f"  {b['product1']} + {b['product2']}")
                print(f"      Support: {b['support']} co-purchases, Bundle price: ${b['bundle_price']}")
            return bundles

    # ==================== NEW: INFLUENCE & NETWORK ANALYSIS ====================

    def get_influential_users(self, limit=10):
        """Find influential users based on purchase pattern propagation (PageRank-like)."""
        with self.driver.session() as session:
            result = session.run("""
                // Users who purchased products BEFORE many others (early adopters)
                MATCH (u:User)-[p1:PURCHASED]->(prod:Product)<-[p2:PURCHASED]-(follower:User)
                WHERE p1.date < p2.date AND u <> follower
                WITH u, COUNT(DISTINCT follower) AS followers,
                     COUNT(DISTINCT prod) AS products_led

                // Factor in rating influence
                OPTIONAL MATCH (u)-[r:RATED]->(rated:Product)
                WITH u, followers, products_led, AVG(r.rating) AS avg_rating

                // Calculate influence score
                WITH u, followers, products_led,
                     COALESCE(avg_rating, 3) AS avg_rating,
                     followers * products_led AS influence_score

                RETURN u.name AS user, u.persona AS persona,
                       followers, products_led,
                       round(avg_rating * 100) / 100 AS avg_rating,
                       influence_score
                ORDER BY influence_score DESC
                LIMIT $limit
            """, limit=limit)

            users = list(result)
            print("\nMost Influential Users (Early Adopters):")
            print("-" * 60)
            if not users:
                print("  No influence data found.")
            for u in users:
                print(f"  {u['user']} ({u['persona']})")
                print(f"      Followers: {u['followers']}, Products led: {u['products_led']}, "
                      f"Avg rating: {u['avg_rating']}, Score: {u['influence_score']}")
            return users

    def get_user_communities(self, min_overlap=3, limit=5):
        """Find user communities based on purchase overlap."""
        with self.driver.session() as session:
            # Find users with significant purchase overlap
            result = session.run("""
                MATCH (u1:User)-[:PURCHASED]->(p:Product)<-[:PURCHASED]-(u2:User)
                WHERE u1.id < u2.id
                WITH u1, u2, COUNT(DISTINCT p) AS shared_products,
                     COLLECT(DISTINCT p.name)[0..5] AS sample_products
                WHERE shared_products >= $min_overlap

                RETURN u1.name AS user1, u1.persona AS persona1,
                       u2.name AS user2, u2.persona AS persona2,
                       shared_products, sample_products
                ORDER BY shared_products DESC
                LIMIT $limit
            """, min_overlap=min_overlap, limit=limit)

            communities = list(result)
            print(f"\nUser Communities (min {min_overlap} shared products):")
            print("-" * 60)
            if not communities:
                print("  No significant communities found.")
            for c in communities:
                print(f"  {c['user1']} ({c['persona1']}) <-> {c['user2']} ({c['persona2']})")
                print(f"      Shared: {c['shared_products']} products - {c['sample_products']}")
            return communities

    # ==================== NEW: ANOMALY DETECTION ====================

    def detect_rating_anomalies(self, limit=10):
        """Detect users with unusual rating patterns (potential fake reviews)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[r:RATED]->(p:Product)
                WITH u,
                     COUNT(r) AS rating_count,
                     AVG(r.rating) AS avg_rating,
                     STDEV(r.rating) AS rating_stddev,
                     COLLECT(r.rating) AS ratings
                WHERE rating_count >= 3

                // Flag users with suspiciously uniform ratings
                WITH u, rating_count, avg_rating,
                     COALESCE(rating_stddev, 0) AS rating_stddev, ratings,
                     // All same rating is suspicious
                     CASE WHEN COALESCE(rating_stddev, 0) < 0.5 THEN 1 ELSE 0 END AS uniform_flag,
                     // Extreme average (all 5s or all 1s) is suspicious
                     CASE WHEN avg_rating > 4.5 OR avg_rating < 1.5 THEN 1 ELSE 0 END AS extreme_flag

                WITH u, rating_count, avg_rating, rating_stddev, ratings,
                     uniform_flag + extreme_flag AS anomaly_score
                WHERE anomaly_score > 0

                RETURN u.name AS user, u.persona AS persona,
                       rating_count,
                       round(avg_rating * 100) / 100 AS avg_rating,
                       round(rating_stddev * 100) / 100 AS rating_stddev,
                       anomaly_score
                ORDER BY anomaly_score DESC, rating_count DESC
                LIMIT $limit
            """, limit=limit)

            anomalies = list(result)
            print("\nPotential Rating Anomalies (Review Quality Check):")
            print("-" * 60)
            if not anomalies:
                print("  No anomalies detected.")
            for a in anomalies:
                print(f"  {a['user']} ({a['persona']})")
                print(f"      Ratings: {a['rating_count']}, Avg: {a['avg_rating']}, "
                      f"StdDev: {a['rating_stddev']}, Anomaly score: {a['anomaly_score']}")
            return anomalies

    def detect_sudden_popularity(self, days=30, threshold=2.0, limit=5):
        """Detect products with sudden spikes in purchases (potential manipulation)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User)-[p:PURCHASED]->(prod:Product)
                WITH prod,
                     SUM(CASE WHEN p.date >= '2024-10-01' THEN 1 ELSE 0 END) AS recent,
                     SUM(CASE WHEN p.date < '2024-10-01' AND p.date >= '2024-07-01' THEN 1 ELSE 0 END) AS previous
                WHERE previous > 0 AND recent > 0
                WITH prod, recent, previous,
                     toFloat(recent) / previous AS growth_rate
                WHERE growth_rate >= $threshold

                RETURN prod.name AS product, prod.price AS price,
                       recent AS recent_purchases,
                       previous AS previous_purchases,
                       round(growth_rate * 100) / 100 AS growth_rate
                ORDER BY growth_rate DESC
                LIMIT $limit
            """, threshold=threshold, limit=limit)

            products = list(result)
            print(f"\nProducts with Sudden Popularity Spike (>{threshold}x growth):")
            print("-" * 60)
            if not products:
                print("  No sudden popularity spikes detected.")
            for p in products:
                print(f"  {p['product']} (${p['price']})")
                print(f"      Recent: {p['recent_purchases']}, Previous: {p['previous_purchases']}, "
                      f"Growth: {p['growth_rate']}x")
            return products

    # ==================== NEW: PERSONALIZED HYBRID ====================

    def get_hybrid_recommendations(self, user_name, limit=5):
        """
        Hybrid recommendation combining multiple signals:
        - Collaborative filtering (what similar users bought)
        - Content-based (category preferences)
        - Engagement (view time, wishlists)
        - Social proof (ratings, popularity)
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (user:User {name: $user_name})

                // Get user's category preferences
                OPTIONAL MATCH (user)-[:PURCHASED]->(:Product)-[:BELONGS_TO]->(pref_cat:Category)
                WITH user, COLLECT(DISTINCT pref_cat.name) AS preferred_categories

                // Find candidate products
                MATCH (rec:Product)-[:BELONGS_TO]->(cat:Category)
                WHERE NOT (user)-[:PURCHASED]->(rec)

                // Collaborative signal
                OPTIONAL MATCH (user)-[:PURCHASED]->(:Product)<-[:PURCHASED]-(sim:User)-[:PURCHASED]->(rec)
                WHERE sim <> user
                WITH user, rec, cat, preferred_categories,
                     COUNT(DISTINCT sim) AS collab_score

                // Content signal (category match)
                WITH user, rec, cat, preferred_categories, collab_score,
                     CASE WHEN cat.name IN preferred_categories THEN 2 ELSE 0 END AS content_score

                // Engagement signal (views, wishlists)
                OPTIONAL MATCH (user)-[v:VIEWED]->(rec)
                OPTIONAL MATCH (user)-[:WISHLISTED]->(rec)
                WITH user, rec, cat, collab_score, content_score,
                     CASE WHEN v IS NOT NULL THEN v.view_count ELSE 0 END AS view_score,
                     CASE WHEN (user)-[:WISHLISTED]->(rec) THEN 3 ELSE 0 END AS wishlist_score

                // Social proof signal
                OPTIONAL MATCH (:User)-[r:RATED]->(rec)
                WITH rec, cat, collab_score, content_score, view_score, wishlist_score,
                     COALESCE(AVG(r.rating), 0) AS avg_rating,
                     COUNT(r) AS rating_count

                // Combine scores
                WITH rec, cat,
                     collab_score * 2 + content_score + view_score + wishlist_score +
                     (avg_rating * rating_count / 10) AS hybrid_score,
                     collab_score, content_score, view_score, wishlist_score,
                     avg_rating, rating_count
                WHERE hybrid_score > 0

                RETURN rec.name AS product, rec.price AS price, cat.name AS category,
                       round(hybrid_score * 100) / 100 AS score,
                       collab_score, content_score, view_score, wishlist_score,
                       round(avg_rating * 100) / 100 AS avg_rating, rating_count
                ORDER BY hybrid_score DESC
                LIMIT $limit
            """, user_name=user_name, limit=limit)

            recommendations = list(result)
            print(f"\nHybrid Recommendations for {user_name}:")
            print("-" * 60)
            if not recommendations:
                print("  No recommendations found.")
            for rec in recommendations:
                print(f"  {rec['product']} (${rec['price']}) - {rec['category']}")
                print(f"      Score: {rec['score']} (collab:{rec['collab_score']}, "
                      f"content:{rec['content_score']}, view:{rec['view_score']}, "
                      f"wish:{rec['wishlist_score']})")
                if rec['rating_count'] > 0:
                    print(f"      Rating: {rec['avg_rating']} ({rec['rating_count']} reviews)")
            return recommendations

    # ==================== NEW: PATH ANALYSIS ====================

    def get_purchase_journey(self, user_name):
        """Analyze user's purchase journey over time."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (u:User {name: $user_name})-[p:PURCHASED]->(prod:Product)-[:BELONGS_TO]->(cat:Category)
                WITH u, p, prod, cat
                ORDER BY p.date
                WITH u,
                     COLLECT({
                         date: p.date,
                         product: prod.name,
                         category: cat.name,
                         price: prod.price
                     }) AS journey
                RETURN journey
            """, user_name=user_name)

            record = result.single()
            if not record:
                print(f"\nNo purchase journey found for {user_name}")
                return []

            journey = record['journey']
            print(f"\nPurchase Journey for {user_name}:")
            print("-" * 60)

            total_spent = 0
            categories_over_time = []
            for step in journey:
                print(f"  {step['date']}: {step['product']} ({step['category']}) - ${step['price']}")
                total_spent += step['price']
                categories_over_time.append(step['category'])

            print(f"\n  Total spent: ${round(total_spent, 2)}")
            print(f"  Category progression: {' -> '.join(categories_over_time)}")
            return journey


def main():
    URI = "bolt://localhost:7687"
    USER = "neo4j"
    PASSWORD = "airspace"

    print("=" * 70)
    print("Neo4j Recommendation Engine PoC v2 - Enhanced Features")
    print("=" * 70)

    try:
        engine = RecommendationEngineV2(URI, USER, PASSWORD)

        # Setup
        engine.clear_database()
        engine.create_enhanced_sample_data(num_users=100)
        engine.show_graph_stats()

        # Get sample users
        with engine.driver.session() as session:
            result = session.run("""
                MATCH (u:User)
                RETURN u.name AS name
                ORDER BY rand()
                LIMIT 3
            """)
            sample_users = [r['name'] for r in result]

        print(f"\nDemo users: {sample_users}")

        # === SECTION 1: ORIGINAL ALGORITHMS ===
        print("\n" + "=" * 70)
        print("SECTION 1: CLASSIC RECOMMENDATION ALGORITHMS")
        print("=" * 70)

        engine.get_collaborative_recommendations(sample_users[0])
        engine.get_category_recommendations(sample_users[0])

        # === SECTION 2: ENGAGEMENT-BASED ===
        print("\n" + "=" * 70)
        print("SECTION 2: ENGAGEMENT & CONVERSION ANALYSIS")
        print("=" * 70)

        engine.get_view_to_purchase_funnel(sample_users[0])
        engine.get_wishlist_recommendations(sample_users[0])

        # === SECTION 3: MARKET BASKET ===
        print("\n" + "=" * 70)
        print("SECTION 3: MARKET BASKET ANALYSIS")
        print("=" * 70)

        # Get a popular product
        with engine.driver.session() as session:
            popular = session.run("""
                MATCH (p:Product)<-[:PURCHASED]-()
                WITH p, COUNT(*) AS purchase_count
                RETURN p.name AS name
                ORDER BY purchase_count DESC
                LIMIT 1
            """).single()

        if popular:
            engine.get_frequently_bought_together(popular['name'])
        engine.get_bundle_suggestions('Electronics')

        # === SECTION 4: NETWORK ANALYSIS ===
        print("\n" + "=" * 70)
        print("SECTION 4: USER NETWORK & INFLUENCE ANALYSIS")
        print("=" * 70)

        engine.get_influential_users(limit=5)
        engine.get_user_communities(min_overlap=3, limit=5)

        # === SECTION 5: ANOMALY DETECTION ===
        print("\n" + "=" * 70)
        print("SECTION 5: ANOMALY DETECTION (FRAUD PATTERNS)")
        print("=" * 70)

        engine.detect_rating_anomalies(limit=5)
        engine.detect_sudden_popularity(days=90, threshold=1.5, limit=5)

        # === SECTION 6: ADVANCED HYBRID ===
        print("\n" + "=" * 70)
        print("SECTION 6: HYBRID PERSONALIZATION")
        print("=" * 70)

        engine.get_hybrid_recommendations(sample_users[0])

        # === SECTION 7: USER JOURNEY ===
        print("\n" + "=" * 70)
        print("SECTION 7: USER JOURNEY ANALYSIS")
        print("=" * 70)

        engine.get_purchase_journey(sample_users[0])

        engine.close()
        print("\n" + "=" * 70)
        print("PoC v2 completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure Neo4j is running with correct credentials.")


if __name__ == "__main__":
    main()

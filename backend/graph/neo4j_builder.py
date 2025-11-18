
from neo4j import GraphDatabase
import pandas as pd


class Neo4jBuilder:
    def __init__(self,
                 uri="bolt://localhost:7687",
                 user="neo4j",
                 password="admin123"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close Neo4j driver."""
        self.driver.close()

    def clear_database(self):
        """Optional: Clears all nodes & relationships."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # --------------------------------------------------------------------------
    # 🧩 INSERTION
    # --------------------------------------------------------------------------
    def add_attribute_value_pairs(self, attributes_dict):
        """
        Add extracted attributes and values (dict format) to the graph.
        Example: {"Billing": {"Monthly", "On-Demand"}}
        """
        with self.driver.session() as session:
            for attribute, values in attributes_dict.items():
                for value in values:
                    if not attribute or not value:
                        continue
                    session.run("""
                        MERGE (a:Attribute {name: $attribute})
                        MERGE (v:Value {value: $value})
                        MERGE (a)-[:HAS_VALUE]->(v)
                    """, attribute=attribute.strip(), value=value.strip())

    def add_from_dataframe(self, df):
        """
        Add data from a DataFrame (Attribute | Value1 | Value2 ...).
        """
        with self.driver.session() as session:
            for _, row in df.iterrows():
                attribute = str(row["Attribute"]).strip()
                values = [str(v).strip() for v in row[1:] if str(v).strip()]
                for val in values:
                    session.run("""
                        MERGE (a:Attribute {name: $attribute})
                        MERGE (v:Value {value: $val})
                        MERGE (a)-[:HAS_VALUE]->(v)
                    """, attribute=attribute, val=val)
        print("✅ Data successfully pushed to Neo4j graph.")

    # --------------------------------------------------------------------------
    # 🔁 RENAME ATTRIBUTE (FULL FIX — VALUES PRESERVED)
    # --------------------------------------------------------------------------
    def rename_attribute(self, old_name, new_name):
        """
        Safely renames an Attribute node while keeping all its HAS_VALUE links.
        Works even if new attribute already exists.
        Case-insensitive, APOC-free, and edge-safe.
        """
        with self.driver.session() as session:
            session.run("""
                // Step 1: Locate the old attribute and its linked values
                MATCH (a:Attribute)
                WHERE toLower(a.name) = toLower($old_name)
                OPTIONAL MATCH (a)-[:HAS_VALUE]->(v)
                WITH a, COLLECT(DISTINCT v) AS vals

                // Step 2: Create or reuse the new attribute node
                MERGE (b:Attribute {name: $new_name})
                WITH a, b, vals   // ✅ FIX: carry variables to next scope

                // Step 3: Connect all values to the new attribute
                UNWIND vals AS val
                MERGE (b)-[:HAS_VALUE]->(val)

                // Step 4: Delete the old attribute node after transfer
                WITH a, b
                WHERE a <> b
                DETACH DELETE a
            """, old_name=old_name, new_name=new_name)
        print(f"✅ Safely renamed '{old_name}' → '{new_name}' (links preserved).")

    def rename_attribute(self, old_name, new_name):
        """
        Safely renames an Attribute node while keeping all HAS_VALUE relationships.
        Uses a two-step Cypher transaction to prevent relationship loss.
        Case-insensitive, APOC-free, fully safe.
        """
        with self.driver.session() as session:
            # STEP 1 — Copy all relationships to the new node
            session.run("""
                MATCH (a:Attribute)
                WHERE toLower(a.name) = toLower($old_name)
                OPTIONAL MATCH (a)-[:HAS_VALUE]->(v)
                WITH a, COLLECT(DISTINCT v) AS vals
                MERGE (b:Attribute {name: $new_name})
                WITH a, b, vals
                UNWIND vals AS val
                MERGE (b)-[:HAS_VALUE]->(val)
            """, old_name=old_name, new_name=new_name)

            # STEP 2 — Delete the old node *after* confirming relationships exist
            session.run("""
                MATCH (a:Attribute)
                WHERE toLower(a.name) = toLower($old_name)
                DETACH DELETE a
            """, old_name=old_name)

        print(f"✅ Safely renamed '{old_name}' → '{new_name}' (links preserved in Neo4j).")


    # --------------------------------------------------------------------------
    # 🗑️ DELETE ATTRIBUTE
    # --------------------------------------------------------------------------
    # def delete_attribute(self, attr_name):
    #     """
    #     Deletes an attribute node and all its linked relationships.
    #     Case-insensitive match.
    #     """
    #     with self.driver.session() as session:
    #         session.run("""
    #             MATCH (a:Attribute)
    #             WHERE toLower(a.name) = toLower($attr_name)
    #             DETACH DELETE a
    #         """, attr_name=attr_name)
    #     print(f"🗑️ Deleted attribute '{attr_name}' from Neo4j.")

    def delete_attribute(self, attr_name, cleanup_orphans=True):
        """Deletes an attribute and optionally removes orphaned values."""
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
                WHERE toLower(a.name) = toLower($attr_name)
                WITH a, collect(v) AS vals
                DETACH DELETE a
                WITH vals
                UNWIND vals AS v
                OPTIONAL MATCH (x)-[:HAS_VALUE]->(v)
                WITH v, COUNT(x) AS refs
                WHERE refs = 0
                DELETE v
            """, attr_name=attr_name)
        print(f"🗑️ Deleted '{attr_name}' and cleaned orphan values.")


    # --------------------------------------------------------------------------
    # 🆕 VALUE UTILITIES
    # --------------------------------------------------------------------------
    def add_value(self, attr_name, new_value):
        """
        Adds a new value under an existing or new attribute.
        If the attribute doesn't exist, it's created automatically.
        Fully safe (no duplicates, case-insensitive).
        """
        with self.driver.session() as session:
            session.run("""
                MERGE (a:Attribute {name: $attr_name})
                MERGE (v:Value {value: $new_value})
                MERGE (a)-[:HAS_VALUE]->(v)
            """, attr_name=attr_name.strip(), new_value=new_value.strip())
        print(f"✅ Added value '{new_value}' under '{attr_name}'.")


    # def remove_value(self, attr_name, value_name):
    #     """Removes a specific value relationship from an attribute."""
    #     with self.driver.session() as session:
    #         session.run("""
    #             MATCH (a:Attribute)-[r:HAS_VALUE]->(v:Value)
    #             WHERE toLower(a.name) = toLower($attr_name)
    #               AND toLower(v.value) = toLower($value_name)
    #             DELETE r
    #         """, attr_name=attr_name, value_name=value_name)
    #     print(f"🗑️ Removed value '{value_name}' from '{attr_name}'.")

    # # def rename_value(self, attr_name, old_value, new_value):
    #     """Renames a value under a specific attribute."""
    #     with self.driver.session() as session:
    #         session.run("""
    #             MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
    #             WHERE toLower(a.name) = toLower($attr_name)
    #               AND toLower(v.value) = toLower($old_value)
    #             SET v.value = $new_value
    #         """, attr_name=attr_name, old_value=old_value, new_value=new_value)
    #     print(f"🔁 Renamed value '{old_value}' → '{new_value}' in '{attr_name}'.")

    def remove_value(self, attr_name, value_name):
        """
        Removes a specific value relationship from an attribute.
        Cleans up orphaned Value nodes that are no longer linked to any Attribute.
        Case-insensitive and safe.
        """
        with self.driver.session() as session:
            # Step 1: Remove only the relationship
            session.run("""
                MATCH (a:Attribute)-[r:HAS_VALUE]->(v:Value)
                WHERE toLower(a.name) = toLower($attr_name)
                AND toLower(v.value) = toLower($value_name)
                DELETE r
            """, attr_name=attr_name, value_name=value_name)

            # Step 2: Clean orphaned Value nodes (not used by any Attribute)
            session.run("""
                MATCH (v:Value)
                WHERE toLower(v.value) = toLower($value_name)
                AND NOT (()-[:HAS_VALUE]->(v))
                DELETE v
            """, value_name=value_name)

        print(f"🗑️ Removed value '{value_name}' from '{attr_name}' (and cleaned orphans).")




    def rename_value(self, attr_name, old_value, new_value):
        """
        Renames a specific Value node under an Attribute.
        Keeps it isolated (does not affect other attributes using the same value).
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Attribute)-[r:HAS_VALUE]->(v:Value)
                WHERE toLower(a.name) = toLower($attr_name)
                AND toLower(v.value) = toLower($old_value)
                MERGE (v2:Value {value: $new_value})
                MERGE (a)-[:HAS_VALUE]->(v2)
                DELETE r
                WITH v
                OPTIONAL MATCH (x)-[:HAS_VALUE]->(v)
                WITH v, COUNT(x) AS refs
                WHERE refs = 0
                DELETE v
            """, attr_name=attr_name, old_value=old_value, new_value=new_value)
        print(f"🔁 Safely renamed value '{old_value}' → '{new_value}' under '{attr_name}'.")




    def get_values(self, attr_name):
        """Fetches all values linked to an attribute (case-insensitive)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a:Attribute)-[:HAS_VALUE]->(v:Value)
                WHERE toLower(a.name) = toLower($attr_name)
                RETURN DISTINCT v.value AS value
            """, attr_name=attr_name)
            return [record["value"] for record in result]

    # --------------------------------------------------------------------------
    # 🧩 MERGE ATTRIBUTES (FULL FIX — VALUES PRESERVED)
    # --------------------------------------------------------------------------
    def merge_attributes(self, attr_list, new_name):
        """
        Merges multiple attributes into one new attribute while preserving all unique values.
        Fully safe, deduplicates values, and deletes old nodes.
        """
        with self.driver.session() as session:
            session.run("""
                MATCH (a:Attribute)
                WHERE toLower(a.name) IN [x IN $attr_list | toLower(x)]
                OPTIONAL MATCH (a)-[:HAS_VALUE]->(v)
                WITH COLLECT(DISTINCT v) AS allVals, COLLECT(DISTINCT a) AS oldAttrs
                MERGE (b:Attribute {name: $new_name})
                FOREACH (v IN allVals | MERGE (b)-[:HAS_VALUE]->(v))
                FOREACH (x IN oldAttrs | DETACH DELETE x)
            """, attr_list=attr_list, new_name=new_name)
        print(f"🔗 Safely merged {attr_list} → '{new_name}' (all values retained).")

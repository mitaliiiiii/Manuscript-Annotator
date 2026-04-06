class ReadingOrderSorter:
    def __init__(self):
        pass

    def sort(self, regions):
        """
        Sorts regions in a human-friendly reading order.
        1. Separate Marginalia from Main Text.
        2. Sort Main Text Top-to-Bottom.
        3. Sort Marginalia based on their Y-position relative to main blocks.
        """
        if not regions:
            return []

        # Category 2 is marginalia/notes
        marginalia = [r for r in regions if r.get("category_id") == 2]
        main_content = [r for r in regions if r.get("category_id") != 2]

        # Sort Main Content: Primary Top-to-Bottom, secondary Left-to-Right
        # We'll use a small Y-tolerance (e.g., 20 pixels) to avoid noise from slight tilts
        main_content_sorted = sorted(main_content, key=lambda r: (r["bbox"][1] // 20, r["bbox"][0]))

        # Sort Marginalia
        marginalia_sorted = sorted(marginalia, key=lambda r: (r["bbox"][1], r["bbox"][0]))

        # Return combined (Main Text first, then Marginalia at the end)
        return main_content_sorted + marginalia_sorted

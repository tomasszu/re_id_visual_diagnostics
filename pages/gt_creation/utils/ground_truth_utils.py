from ground_truth import assign_event_to_gt, create_new_gt_id, clear_gt_assignment


def resolve_gt(row):
    gt = row.get("ground_truth")

    if isinstance(gt, dict):
        return gt.get("gt_vehicle_id")

    return None


def merge_gt(storage, df, query_row, cand_row):
    query_gt = resolve_gt(query_row)
    cand_gt = resolve_gt(cand_row)

    # ---------------- CASE 1: neither assigned ----------------
    if not query_gt and not cand_gt:
        new_gt = create_new_gt_id()

        assign_event_to_gt(storage, query_row["obj_key"], new_gt)
        assign_event_to_gt(storage, cand_row["obj_key"], new_gt)
        return

    # ---------------- CASE 2: one side assigned ----------------
    if query_gt and not cand_gt:
        assign_event_to_gt(storage, cand_row["obj_key"], query_gt)
        return

    if cand_gt and not query_gt:
        assign_event_to_gt(storage, query_row["obj_key"], cand_gt)
        return

    # ---------------- CASE 3: merge clusters ----------------
    if query_gt and cand_gt and query_gt != cand_gt:
        for _, r in df.iterrows():
            r_gt = resolve_gt(r)

            if r_gt == cand_gt:
                assign_event_to_gt(storage, r["obj_key"], query_gt)

def remove_events_from_gt(storage, rows):
    for _, row in rows.iterrows():
        clear_gt_assignment(storage, row["obj_key"])


def split_events_to_new_gt(storage, rows):
    new_gt = create_new_gt_id()

    for _, row in rows.iterrows():
        assign_event_to_gt(storage, row["obj_key"], new_gt)

    return new_gt
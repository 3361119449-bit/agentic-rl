"""Compare the repaired booking tasks without changing actual reservations."""


def repaired_db_mode(task_id: str) -> str | None:
    return {"14": "payment_order", "23": "booking_order"}.get(str(task_id))


def booking_comparison_constructor(constructor, mode: str | None):
    if mode is None:
        return constructor
    if mode not in {"payment_order", "booking_order"}:
        raise ValueError(f"unknown booking DB comparison: {mode}")

    def construct(*args, **kwargs):
        from tau2.utils.utils import get_dict_hash

        environment = constructor(*args, **kwargs)
        original_ids = frozenset(environment.tools.db.reservations)

        def get_db_hash():
            state = environment.tools.db.model_dump()
            reservations = state["reservations"]
            new_ids = reservations.keys() - original_ids
            for reservation_id in new_ids:
                reservations[reservation_id]["payment_history"].sort(
                    key=lambda payment: (payment["payment_id"], payment["amount"])
                )
            if mode == "booking_order":
                bookings = []
                for reservation_id in new_ids:
                    booking = reservations.pop(reservation_id)
                    booking.pop("reservation_id")
                    bookings.append((get_dict_hash(booking), reservation_id, booking))
                canonical_ids = {}
                for index, (_, reservation_id, booking) in enumerate(sorted(bookings)):
                    canonical_id = f"new-booking:{index}"
                    if canonical_id in reservations:
                        raise ValueError("canonical booking ID collides with an existing ID")
                    canonical_ids[reservation_id] = canonical_id
                    reservations[canonical_id] = {**booking, "reservation_id": canonical_id}
                for user in state["users"].values():
                    ids = user["reservations"]
                    ordered = iter(
                        sorted(canonical_ids[rid] for rid in ids if rid in canonical_ids)
                    )
                    user["reservations"] = [
                        next(ordered) if rid in canonical_ids else rid for rid in ids
                    ]
            return get_dict_hash(state)

        environment.get_db_hash = get_db_hash
        return environment

    return construct

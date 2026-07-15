import { submitOrder } from "../api/orders";

export function canSubmit(status: string, hasPermission: boolean): boolean {
  return status === "PENDING" && hasPermission;
}

export async function submitPendingOrder(
  orderId: string,
  status: string,
  hasPermission: boolean,
): Promise<void> {
  if (!canSubmit(status, hasPermission)) {
    throw new Error("ORDER_SUBMIT_FORBIDDEN");
  }
  await submitOrder(orderId);
}

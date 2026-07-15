export async function submitOrder(orderId: string): Promise<void> {
  const response = await fetch(`/api/orders/${orderId}/submit`, { method: "POST" });
  if (!response.ok) {
    throw new Error("ORDER_SUBMIT_FAILED");
  }
}

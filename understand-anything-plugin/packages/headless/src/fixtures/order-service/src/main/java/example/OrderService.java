package example;

public class OrderService {
    public void submit(String orderId, boolean canSubmit) {
        if (!canSubmit) {
            throw new SecurityException("ORDER_SUBMIT permission is required");
        }
        persist(orderId);
    }

    private void persist(String orderId) {
        System.out.println(orderId);
    }
}

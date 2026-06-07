import 'alert_service.dart';
import '../models/alert_model.dart';

class MockAlertService extends AlertService {
  @override
  Future<List<Alert>> getAlerts() async {
    await Future.delayed(const Duration(seconds: 1));

    return [
      Alert(id: "1", type: "Fire", severity: "High", lat: 12.97, lon: 77.59),
      Alert(id: "2", type: "Flood", severity: "Medium", lat: 13.00, lon: 77.60),
    ];
  }
}

import '../models/report_model.dart';

import 'report_service.dart';

class MockReportService implements ReportService {
  @override
  Future<void> submitReport(ReportModel report) async {
    await Future.delayed(const Duration(seconds: 2));

    print("Mock Report Submitted: ${report.description}");
  }
}

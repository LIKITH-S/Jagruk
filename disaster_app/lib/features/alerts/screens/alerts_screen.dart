import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../bloc/alert_bloc.dart';
import '../bloc/alert_state.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  Color getColor(String severity) {
    switch (severity.toLowerCase()) {
      case 'high':
        return Colors.red;

      case 'medium':
        return Colors.orange;

      default:
        return Colors.green;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Alerts")),

      body: BlocBuilder<AlertBloc, AlertState>(
        builder: (context, state) {
          if (state is AlertLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          if (state is AlertLoaded) {
            return ListView.builder(
              itemCount: state.alerts.length,

              itemBuilder: (context, index) {
                final alert = state.alerts[index];

                return Card(
                  margin: const EdgeInsets.all(12),

                  child: ListTile(
                    leading: Icon(
                      Icons.warning,

                      color: getColor(alert.severity),
                    ),

                    title: Text(alert.type),

                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,

                      children: [
                        Text("Severity: ${alert.severity}"),

                        Text("Lat: ${alert.location.lat}"),

                        Text("Lon: ${alert.location.lon}"),
                      ],
                    ),
                  ),
                );
              },
            );
          }

          if (state is AlertError) {
            return Center(child: Text(state.message));
          }

          return const SizedBox();
        },
      ),
    );
  }
}

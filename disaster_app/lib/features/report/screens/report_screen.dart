import 'package:flutter/material.dart';

import 'package:flutter_bloc/flutter_bloc.dart';

import '../bloc/report_bloc.dart';
import '../bloc/report_event.dart';
import '../bloc/report_state.dart';

class ReportScreen extends StatefulWidget {
  const ReportScreen({super.key});

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  final descriptionController = TextEditingController();

  String selectedDisaster = "Fire";

  final disasterTypes = ["Fire", "Flood", "Earthquake", "Cyclone"];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Report Disaster")),

      body: BlocConsumer<ReportBloc, ReportState>(
        listener: (context, state) {
          if (state is ReportSuccess) {
            ScaffoldMessenger.of(
              context,
            ).showSnackBar(const SnackBar(content: Text("Report Submitted")));
          }

          if (state is ReportError) {
            ScaffoldMessenger.of(
              context,
            ).showSnackBar(SnackBar(content: Text(state.message)));
          }
        },

        builder: (context, state) {
          return SingleChildScrollView(
            padding: const EdgeInsets.all(16),

            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,

              children: [
                const Text(
                  "Disaster Type",

                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),

                const SizedBox(height: 10),

                DropdownButtonFormField(
                  initialValue: selectedDisaster,

                  items: disasterTypes.map((type) {
                    return DropdownMenuItem(value: type, child: Text(type));
                  }).toList(),

                  onChanged: (value) {
                    setState(() {
                      selectedDisaster = value!;
                    });
                  },
                ),

                const SizedBox(height: 20),

                const Text(
                  "Description",

                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),

                const SizedBox(height: 10),

                TextField(
                  controller: descriptionController,

                  maxLines: 5,

                  decoration: const InputDecoration(
                    border: OutlineInputBorder(),

                    hintText: "Describe the disaster...",
                  ),
                ),

                const SizedBox(height: 30),

                SizedBox(
                  width: double.infinity,

                  child: ElevatedButton(
                    onPressed: state is ReportLoading
                        ? null
                        : () {
                            context.read<ReportBloc>().add(
                              SubmitReport(
                                description: descriptionController.text,

                                disasterType: selectedDisaster,

                                lat: 12.9716,

                                lon: 77.5946,
                              ),
                            );
                          },

                    child: state is ReportLoading
                        ? const Padding(
                            padding: EdgeInsets.all(8),

                            child: CircularProgressIndicator(),
                          )
                        : const Padding(
                            padding: EdgeInsets.all(14),

                            child: Text("Submit Report"),
                          ),
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }
}
